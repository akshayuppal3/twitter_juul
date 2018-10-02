####################
##Class containing##
#helper functions###
####################
from time import sleep
import tweepy
import os

logdir =  os.path.abspath("../output/hexagon/")
format = "%(asctime)-15s      %(message)s"

def output_to_csv(df,filename):
    df.to_csv(path_or_buf=filename)

#handle the rate limit (wait for 15min (API constarints))
def limit_handler(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            print("sleeping -rate limit exceeded")
            sleep(15)