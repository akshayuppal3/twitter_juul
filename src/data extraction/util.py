####################
##Class containing##
# helper functions###
####################
from time import sleep
import tweepy
import os
import argparse

logdir = os.path.abspath("../../output/hexagon/")
twintDir = os.path.abspath("../../output/twintData")
inputdir = os.path.abspath("../../input/")
format = "%(asctime)-15s      %(message)s"
dateFormat = "%Y-%m-%d"
testLimit = 5
twintLimit = 1
userTimelineLimit = 200 # limit for the no of tweets extracted from user timeline
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
def getUsers(df,type):
	if (df is not None and not df.empty):
		if (type == 'ID'):
			if ('userID' in df):
				unique_users = list(df['userID'].unique())
				return unique_users
			else:
				return None
		elif(type == 'name'):
			if ('userName' in df):
				unique_names = list(df['userName'].unique())
				return unique_names
			else:
				return None
	else:
		return None