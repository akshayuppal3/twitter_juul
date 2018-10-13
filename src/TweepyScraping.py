############################
###### Tweepy interface ####
####### for twitter  #######
############################
import pandas as pd
import os
import tweepy
from tweepy import OAuthHandler
from authentication import Authenticate
import util

# constants
hashtags = '#juulvapor OR #juulnation OR #doit4juul OR #juul'
lang = "en"
inceptionDate = "2018-01-08"
retweet_status = 'retweeted_status'
hash = 'hashtags'

## to remove keys from gitignore
class Twitter:
	def __init__(self):
		self.api = self.authorization()

	##@TODO put all the comments with python docstring
	##authorization with twitter app
	def authorization(self):
		ob = Authenticate()
		consumer_key = ob.getConsumerKey()
		consumer_secret = ob.getConsumerSecret()
		access_token = ob.getAccessToken()
		access_secret = ob.getAccessSecret()
		author = OAuthHandler(consumer_key, consumer_secret)
		author.set_access_token(access_token, access_secret)
		# change to accomodate rate limit errors
		api = tweepy.API(author, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
		return (api)

	def getHashtags(self, tweetObj, extended=False):
		if extended == True:
			if 'retweeted_status' in tweetObj._json.keys():
				if len(tweetObj.retweeted_status.entities['hashtags']) != 0:
					hashtags = [i['text'] for i in tweetObj.retweeted_status.entities['hashtags']]
				else:
					hashtags = None
			else:
				hashtags = self.getHashtags(tweetObj,extended=False)
		else:
			if len(tweetObj.entities['hashtags']) != 0:
				hashtags = [j['text'] for j in tweetObj.entities['hashtags']]
			else:
				hashtags = "None"
		return hashtags

	# @params passing tweet, friendOpt and userinfo(in case of following)
	# returns data frame of tweet and user info
	def getTweetObject(self, tweetObj, friendOpt=False, user=None):
		if friendOpt is True:  # in case we dont have friend list
			friendList = [self.api.friends_ids(tweetObj.user.id)]
		else:
			friendList = "None"
		if 'retweeted_status' in tweetObj._json.keys():
			text = tweetObj.retweeted_status.full_text.replace("\n", " ")
			hashtags = self.getHashtags(tweetObj,extended=True)        # for retweeted status
		else:
			text = tweetObj.full_text.replace("\n", " ")
			hashtags = self.getHashtags(tweetObj)
		if user is None:
			data = pd.DataFrame.from_records(
				[{
					'tweetId': tweetObj.id_str,
					'userID': tweetObj.user.id,
					'tweetText': text,
					'tweetCreatedAt': tweetObj.created_at,
					'parentID': 'None',
					'favourites_count': tweetObj.user.favourites_count,
					'userLocation': tweetObj.user.location,
					'userName': tweetObj.user.name,
					'userDescription': tweetObj.user.description.replace("\n", " "),
					'userCreatedAt': tweetObj.user.created_at,
					'imageurl': tweetObj.user.profile_image_url,
					'userFollowersCount': tweetObj.user.followers_count,
					'friendsCount': tweetObj.user.friends_count,
					'friendList': friendList,
					'hashtags': hashtags,
					'retweetCount': tweetObj.retweet_count,
					'retweeted': tweetObj.retweeted,
					'lang': tweetObj.lang,
				}], index=None, coerce_float=False)
		else:
			data = pd.DataFrame.from_records(
				[{
					'tweetId': "None",
					'userID': user.id,
					'parentID': tweetObj.user.id,
					'favourites_count': "None",
					'userName': user.name,
					'userDescription': user.description,
					'userCreatedAt': user.created_at,
					'imageurl': user.profile_image_url,
					'userFollowersCount': user.followers_count,
					'friendsCount': user.friends_count,
					'friendList': "None"
				}], index=[0])
		return data



# function to get twitter and user info and return df
# @params api: api_handler, hashtags: query parameters, inceptionDate: start date for hashtags, lang: for language restriction
def getTwitterData(self, queryParams, inceptionDate, lang="en"):
	df = pd.DataFrame([])
	#     df.astype('object')       # as some data contains list
	try:
		for tweet in util.limit_handler(tweepy.Cursor(self.api.search, q=(queryParams), since_id=inceptionDate,
		                                              lang=lang).items()):  # (lang=en) : lang restriction
			friendList = [friend for friend in
			              util.limit_handler(tweepy.Cursor(self.api.friends_ids,
			                                               id=tweet.user.id).items())]  # items value set only for test
			data = self.getTweetObject(tweet, friendOpt=False)
			df = df.append(data, ignore_index=True)
			for userID in friendList:
				user = self.api.get_user(userID)  # @TODO should be reaplced for statuses lookup(100: batch)
				userData = self.getTweetObject(tweet, friendOpt=False, user=user)
				if user.id not in df.userID:  # prevent duplication in data
					df = df.append(userData, ignore_index=True)
		return df
	except tweepy.TweepError as e:
		print(e.api_code)
		print(e.reason)


def searchTweets(self):
	df = self.getTwitterData(hashtags, inceptionDate, lang)
	if (df):
		df = df.set_index('userID')  # Changing index for mongoDb
		os.chdir('../output')
		util.output_to_csv(df, filename='juulDataset.csv')
