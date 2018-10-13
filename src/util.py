####################
##Class containing##
# helper functions###
####################
from time import sleep
import tweepy
import os
import argparse
import winsound

logdir = os.path.abspath("../output/hexagon/")
format = "%(asctime)-15s      %(message)s"
dateFormat = "%Y-%m-%d"
sound_duration = 1000  # millisecond
sound_freq = 440  # Hz


def output_to_csv(df, filename):
	df.to_csv(path_or_buf=filename,index=None)


# conversion of str to bool
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


# handle the rate limit (wait for 15min (API constarints))
def limit_handler(cursor):
	while True:
		try:
			yield cursor.next()
		except tweepy.RateLimitError:
			print("sleeping -rate limit exceeded")
			sleep(15)


# play a sound when the code completes

def playSound():
	winsound.Beep(sound_duration, sound_freq)
