####################
##Class containing##
# helper functions###
####################
from time import sleep
import tweepy
import os
import argparse

logdir = os.path.abspath("../../output/hexagon/")
inputdir = os.path.abspath("../../input/")
format = "%(asctime)-15s      %(message)s"
dateFormat = "%Y-%m-%d"
testLimit = 5
userTimelineLimit = 200 # limit for the no of tweets extracted from user timeline
startDate = '2018-05-01'
endDate = '2018-05-02'

def output_to_csv(df, filename):
	if (df is not None and not df.empty):
		df.to_csv(path_or_buf=filename,index=None)

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
