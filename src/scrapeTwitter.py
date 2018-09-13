import twint
import pandas as pd
import os
import pandas.io.common
import util
import logging
from logging.handlers import RotatingFileHandler

searchQ = ["#doit4juul", "#juul", "#juulvapor", "#juulnation"]
inceptionDate = "2018-09-09"
logging.basicConfig(level="INFO", format= util.format, filename=(util.logdir + "/twintScrapingLogs.log"))
logger = logging.getLogger("logger")
handler = RotatingFileHandler(util.logdir + "/twintScrapingLogs.log", maxBytes=10000000, backupCount=10)
logger.addHandler(handler)

class ScrapeTwitter:
    def __init__(self):
        self.outputPath = os.path.abspath("../output/twintData/")
        self.userLimit = 1
        self.followingLimit = 1

    def getTweetData(self):
        param = self.twintParam(op="Search")
        df = pandas.DataFrame()
        for idx, element in enumerate(searchQ):
            param.Search = element
            param.Output = str(util.logdir + "\juulMainfile" + str(idx) + ".csv")
            twint.run.Search(param)
            if (os.path.isfile(param.Output)):
                df = self.readCSV(param.Output)
                df.append(df, ignore_index=True)
        if (not df.empty):
            logging.info('[INFO] main tweet files for hashtags created')
        return(df)

    def getUserData(self,df):
        if (not df.empty):
            df.set_index("id")
            ## get the csv data
            df_final = df.drop_duplicates(subset='id', keep="first")
            self.toCSV(df_final, "/juulUserTwint.csv")
            logging.info('[INFO] user data created')
            userList = list(df_final.username)
            logging.info('[INFO] extracted user list')
            return (userList)

    def followingData(self, userList):
        param = self.twintParam(op="Following")
        dfFoll = []
        for idx, friend in enumerate(userList):
            param.Username = friend
            param.Output = str(util.logdir + "/juulFollowing" + str(idx) + ".csv")
            logging.info('[INFO] subfile %s of following created' % param.Output)
            twint.run.Following(param)
            if (os.path.exists(param.Output)):
                dfFoll = pd.read_csv(param.Output)
                dfFoll.assign(Parent=param.Username)     #@TODO add the parent field to the table
                dfFoll.append(dfFoll, ignore_index=True)
        if (not dfFoll.empty):
            self.toCSV(dfFoll, "/juulFollowingTwint.csv")
            logging.info("[INFO] final user list created")

    def getData(self):
        logging.info("[INFO] new extraction process started")
        df = self.getTweetData()
        if (not df.empty):
            userList = self.getUserData(df)
            if (userList):
                self.followingData(userList)

    def twintParam(self, op="Search"):
        config = twint.Config()
        config.Store_csv = True
        if (op == "Search"):
            config.Since = inceptionDate
            config.Store_csv = True
            config.limit = self.userLimit
        elif (op =="Following"):
            config.User_full = True
            config.limit = self.followingLimit
        return (config)

    def execData(self, config, op="Search"):
        if (op == "Search"):
            twint.run.Search(config)
        elif (op == "Following"):
            twint.run.Following(config)

    def toCSV(self, df, filename):
        path = os.path.abspath(util.logdir)
        df.to_csv(path+filename)

    def readCSV(self,path):
        try:
            df = pd.read_csv(path)
            return df
        except FileNotFoundError:
            logging.error("[ERROR] file not found")
        except pd.io.common.EmptyDataError:
            print ("file is empty at %s " % path)
            logging.error("[ERROR] empty file")

if __name__ == '__main__':
    ob = ScrapeTwitter()
    ob.getData()
    logging.info('[INFO] data extraction finished')
