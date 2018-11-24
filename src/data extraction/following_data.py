#############################
##Class for scraping ########
##following data twitter#####
##########version 2 #########
#############################

from authentication import Authenticate
from tweepy import OAuthHandler
import pandas as pd
import util
import argparse
import logging
import tweepy
import os
from tqdm import tqdm
import ast
import time
import math

logging.basicConfig(level="INFO", format= util.format, filename= os.path.join(util.logdir,"followingData.log"))


## just passing the file and it would extract the following data

class twitter_following():

    def __init__(self):
        self.api = self.authorization()

    ## currently returning one API
    ## @TODO need to change for multiple API
    def authorization(self):
        ob = Authenticate()
        consumer_key = ob.getConsumerKey()
        consumer_secret = ob.getConsumerSecret()
        access_token = ob.getAccessToken()
        access_secret = ob.getAccessSecret()
        author = OAuthHandler(consumer_key, consumer_secret)
        author.set_access_token(access_token, access_secret)
        # change to accomodate rate limit errors
        api = tweepy.API(author, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, retry_count=5, retry_delay=5)
        return (api)


    # @param df, filename , testMode(bool)
    # @return None
    # writes to an excel file
    def getFriendsData(self,df,output_path):
        users = util.getUsers(df,type= 'ID')
        try:
            if users:
                for index,user in enumerate(tqdm(users)):
                    try:
                        friendList = self.api.friends_ids(user,
                                                          count=util.friendLimit)  # returns list of friends
                        df = pd.DataFrame({'userID':user,
                                           'following':[friendList]}, index=[0])
                        util.df_write_excel(df,output_path)
                    except:
                        logging.error("Some error in connection")
                        time.sleep(60 * 10)
                        continue
                    finally:
                        print(index)

        except tweepy.TweepError as e:          # except for handling tweepy api call
            print("[Error] " + e.reason)

    # @return DataFrame @ param friends ID and parent ID
    # takes batch of friend ids and returns detailed information of user
    def getFriendBatch(self, friendIds, parent_id):
        api = self.api
        data = pd.DataFrame([])
        try:
            user = api.lookup_users(friendIds, include_entities=True)  # api to look for user (in batch of 100)
            if user:
                for idx, statusObj in enumerate(user):
                    userData = util.getTweetObject(tweetObj=statusObj, parentID=parent_id)
                    data = data.append(userData, ignore_index=True)
                return data
            else:
                print("no user found for the batch")

        except tweepy.TweepError as e:
            print(e.reason)
            logging.error("[Error] " + e.reason)
            # self.getFriendBatch(friendIds, parent_id)       # check when connection is getting lost

        except:
            print("connection timeout")
            logging.error("[lookup users] Some error in api or connection")
            # self.getFriendBatch(friendIds, parent_id)       # check when connection is getting lost

    # return None
    # get the detailed following data for the users and write to excel
    def get_detail_friends_data(self,df,output_path):
        with tqdm(total=(len(list(df.iterrows())))) as pbar:
            for index,row in tqdm(df.iterrows()):
                pbar.update(1)                                     # handling tqdm for pandas
                parent_id = row['userID']
                friends_data = ast.literal_eval(row['following'])   # as data as interpreted as string instead of list
                if len(friends_data) > 100:       # as api.lookup users take data in batch of 100
                    batch_size = int(math.ceil(len(friends_data) / 100))
                    for i in tqdm(range(batch_size)):
                        dfBat = friends_data[(100 * i): (100 * (i + 1))]
                        friends_detailed = self.getFriendBatch(dfBat, parent_id)
                        if friends_detailed is not None:
                            util.df_write_excel(friends_detailed,output_path)   # write the data to excel file
                else:
                    friends_detailed = self.getFriendBatch(friends_data, parent_id)
                    if friends_detailed is not None:
                        util.df_write_excel(friends_detailed,output_path)


if __name__ == '__main__':
    ob = twitter_following()
    parser = argparse.ArgumentParser(description='Extracting data from userDataFile')
    parser.add_argument('-i', '--inputFile', help='Specify the input file path for extracting friends', required=False)
    parser.add_argument('-i2', '--inputFile2', help='Specify the input file path with user and friends id', required=False)
    parser.add_argument('-o',  '--outputFile', help='Specify the output file name with following data',default='followingList')
    parser.add_argument('-o2', '--outputFile2', help='Specify the output file name with following data',default='followingDetailedList')
    args = vars(parser.parse_args())
    if (args['inputFile']):
        logging.info('[NEW] ---------------------------------------------')
        logging.info('new extraction process started')
        filename_input = args['inputFile']
        filename_output = args['outputFile']
        filename_output = os.path.join(util.inputdir,filename_output+'.xlsx')
        df = util.readCSV(filename_input)
        ob.getFriendsData(df,filename_output)
        logging.info("File creation of basic user and following completed")
    if (args['inputFile2']):
        logging.info('[NEW] ---------------------------------------------')
        logging.info('getting detailed following list for users')
        filename_input = args['inputFile2']
        filename_output = args['outputFile2']
        filename_output = os.path.join(util.inputdir,filename_output+'.xlsx')
        df = util.read_excel(filename_input)
        ob.get_detail_friends_data(df,filename_output)
        logging.info("File creation of detailed user completed")





