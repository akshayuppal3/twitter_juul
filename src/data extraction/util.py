####################
##Class containing##
# helper functions###
####################
from time import sleep
import tweepy
import os
import argparse
import pandas as pd
import pandas.io.common
from pathlib import Path
import json

dir_name = os.getcwd()
path1 = Path(os.getcwd()).parent.parent
filepath = os.path.join(path1, 'config.json')
with open(filepath) as f:
    data = json.load(f)
logdir = os.path.join(path1,data['logdir'])
twintDir = os.path.join(path1,data['twintdir'])
inputdir = os.path.join(path1,data['inputdir'])
format = "%(asctime)-15s      %(message)s"
dateFormat = "%Y-%m-%d"
testLimit = 5
twintLimit = 1
userTimelineLimit = 200  # limit for the no of tweets extracted from user timeline
friendLimit = 100
startDate = '2018-05-01'
endDate = '2018-05-02'


def output_to_csv(df, filename):
    if (df is not None and not df.empty):
        df.to_csv(filename, sep=",", line_terminator='\n', index=None)
    else:
        print("datframe is empty")


# conversion of str to bool
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# @Deprecated
# handle the rate limit (wait for 15min (API constarints))
def limit_handler(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            print("sleeping -rate limit exceeded")
            sleep(15)


# @param df
# @return list of unique users
def getUsers(df, type):
    if (df is not None and not df.empty):
        if (type == 'ID'):
            if ('userID' in df):
                unique_users = list(df['userID'].unique())
                return unique_users
            else:
                return None
        elif (type == 'name'):
            if ('userName' in df):
                unique_names = list(df['userName'].unique())
                return unique_names
            else:
                return None
    else:
        return None


##read CSV to generate dataframe
##@return df
def readCSV(path):
    try:
        df = pd.read_csv(path, lineterminator='\n', index_col=0)
        if "userName\r" in df:  # windows problem
            df["userName\r"] = df["userName\r"].str.replace(r'\r', '')
            df.rename(columns={'userName\r': "userName"}, inplace=True)
        return df
    except FileNotFoundError:
        print("[ERROR] file not found")
    except pd.io.common.EmptyDataError:
        print("[ERROR] empty file")
