#############################
###### Extracting data ######
## from hexagon API and then#
## passing to twitter API ###
############################
# @Author : Akshay

from authentication import Authenticate
from TweepyScraping import Twitter
import requests
import urllib.request
import json
import pandas as pd
import util
import os
import math
import argparse
import logging
import tweepy
import datetime
import ast
from tqdm import tqdm

startDate = '2018-03-01'
endDate = '2018-05-02'
monitorID = "11553243040"  # juulMonitor twitter filter ID (numeric field)

logging.basicConfig(level="INFO", format= util.format, filename=(util.logdir + "/hexagonScrapingLogs.log"))
# logger = logging.getLogger("logger")
authenticateURL = "https://api.crimsonhexagon.com/api/authenticate"
baseUrl = "https://api.crimsonhexagon.com/api/monitor"

class Hexagon:
	def __init__(self,testMode = False):
		ob = Twitter()
		self.authenticateURL = authenticateURL
		self.authToken = self.getAuthToken()
		self.baseUrl = baseUrl
		self.api = ob.api
		self.hexagonData = self.getHexagonData(startDate, endDate, testMode= testMode)


	def getAuthToken(self):
		ob = Authenticate()
		username = ob.username
		password = ob.password
		querystring = {
			"username": username,
			"noExpiration": "true",
			"password": password
		}
		try:
			response = requests.request("GET", headers={}, url=self.authenticateURL, params=querystring)
			if (response.status_code == 200):
				result = response.json()
				authToken = result['auth']
				authToken = "&auth=" + authToken
				return (authToken)
		except requests.ConnectionError as e:
			logging.error('[ERROR] %s',e)


	# @paramms startD,endD : type <string>
	def getDates(self,startD,endD):
		dates = "&start=" + startD + "&end=" + endD  # Combines start and end date into format needed for API call
		return dates

	# responsible for splitting a date range to individual months
	# @returns start_dates and end_Dates in str format
	def getDateRange(self,begin,end):
		dt_start = datetime.datetime.strptime(begin,util.dateFormat)
		dt_end =  datetime.datetime.strptime(end,util.dateFormat)
		one_day = datetime.timedelta(1)
		start_dates = [dt_start.strftime(util.dateFormat)]
		end_dates = []
		today = dt_start
		while today < dt_end:
			tomorrow = today + one_day
			if (tomorrow.month != today.month):
				start_dates.append(tomorrow.strftime(util.dateFormat))
				end_dates.append(today.strftime(util.dateFormat))
			today = tomorrow

		end_dates.append(dt_end.strftime(util.dateFormat))
		return (start_dates,end_dates)

	def getEndPoint(self, endpoint):
		return '{}/{}?'.format(self.baseUrl, endpoint)

	def getJsonOb(self,startD,endD):
		webURL = urllib.request.urlopen(self.getURL(startD,endD))
		data = webURL.read().decode('utf8')
		theJSON = json.loads(data)
		return theJSON


	def getURL(self,startD,endD):
		endpoint = self.getEndPoint('posts')
		extendLimit = "&extendLimit=true"  # extends call number from 500 to 10,000
		fullContents = "&fullContents=true"  # Brings back full contents for Blog and Tumblr posts which are usually truncated around sea
		url = '{}id={}{}{}{}{}'.format(endpoint, monitorID, self.authToken, self.getDates(startD,endD), extendLimit, fullContents)
		return url

	# columns for hexagon API object
	def getColumnsData(self, hexagonObj):
		data = pd.DataFrame(
			{
				'tweetID': hexagonObj['url'].split("status/")[1],
				'url': hexagonObj['url'],
				'type': hexagonObj['type'],
				'title': hexagonObj['title'],
				'location': hexagonObj['location'] if 'location' in hexagonObj else "",
				'language': hexagonObj['language']
				# 'contents' : hexagonObj['contents']      #doesn't seem to have from the API call
			}, index=[0]
		)
		return data

	def getJSONData(self,startD,endD,test_mode= False):
		JSON = self.getJsonOb(startD, endD)
		df = pd.DataFrame([])
		data = JSON['posts'][:util.testLimit] if test_mode == True 	else JSON['posts']
		for iter in tqdm(data):
			data = self.getColumnsData(iter)
			df = df.append(data, ignore_index=True)
		return df

	# if data > 10000 true else false
	def checkVolumeData(self, startD, endD):
		JSON = self.getJsonOb(startD, endD)
		if (JSON['totalPostsAvailable'] > 10000):
			return True
		else:
			return False

	# works for data > 10000 (extract month wise data)
	# @returns the hexagon data
	def getHexagonData(self, startD, endD, testMode = False):
		logging.info('[INFO] extraction of Hexagon data started')
		logging.info('[INFO] test mode selected') if testMode == True else None
		df = pd.DataFrame([])
		if (self.checkVolumeData(startD,endD)):
			logging.info('[INFO] Data being extracted in batches')
			startDates, endDates = self.getDateRange(startD,endD)
			for startD,endD in tqdm(zip(startDates,endDates)):
				data = self.getJSONData(startD,endD,testMode)
				df.append(data,ignore_index = True)
		else:
			df = self.getJSONData(startD,endD,testMode)
		logging.info('[INFO] all data extracted from hexagon')
		return df[0:util.testLimit] if (testMode == True) else df   # changes for the test mode

	def getBatchTwitter(self,tweetIDs,parent_id = None,friendOpt = False,test_mode=False):
		data = pd.DataFrame([])
		ob = Twitter()
		api = ob.api
		try:
			user = api.statuses_lookup(tweetIDs,include_entities=True,tweet_mode='extended')  # update as per for full_text
			for idx, statusObj in enumerate(user):
				userData = ob.getTweetObject(tweetObj=statusObj, friendOpt=friendOpt,parentID = parent_id,test_mode=test_mode)
				data = data.append(userData, ignore_index=True)
			return data

		except tweepy.TweepError as e:
			logging.error("[Error] " + e.reason)

	def getFriendBatch(self,friendIds,parent_id,test_mode = False):
		data = pd.DataFrame([])
		ob = Twitter()
		api = ob.api
		try:
			user = api.lookup_users(friendIds, include_entities=True)  # api to look for user (in batch of 100)
			for idx, statusObj in enumerate(user):
				userData = ob.getTweetObject(tweetObj=statusObj, parentID=parent_id, test_mode=test_mode)
				data = data.append(userData, ignore_index=True)
			return data

		except tweepy.TweepError as e:
			logging.error("[Error] " + e.reason)

	# Function to get the friends information
	def getFriendData(self,df_twitter,test_mode = False):
		data = pd.DataFrame([])
		final_data = pd.DataFrame([])
		if not df_twitter.empty:
			for parentId,friendList in tqdm(zip(df_twitter.userID,df_twitter.friendList)):
				if type(friendList) == str:
					friends = ast.literal_eval(friendList)  # as the friend list in the dataframe is string
				else:
					friends = friendList
				if len(friends) != 0:
					if (len(friends) > 100):
						batchSize = int(math.ceil(len(friends)/ 100))
						for i in range(batchSize):
							dfBat = friends[(100* i) : (100 * (i + 1))]
							temp = self.getFriendBatch(dfBat,parent_id = parentId ,test_mode = test_mode)
							data = data.append(temp, ignore_index = True)
					else:
						data = self.getFriendBatch(friends,parent_id = parentId, test_mode = test_mode)
				else:
					continue        # in case of an empty friends list
				final_data = final_data.append(data)         #appending all the friends for a user to the record
		return final_data

	def getUserTimelineData(self,user,test_mode = False):
		ob = Twitter()
		userData = pd.DataFrame([])
		if user is not None:
			count = util.testLimit if test_mode == True else util.userTimelineLimit
			status = self.api.user_timeline(user,count = count, tweet_mode = 'extended')
			for statusObj in status:
				data = ob.getTweetObject(statusObj,test_mode=test_mode)
				userData = userData.append(data)
			return userData

	def getuserTimeline(self,df_twitter,test_mode = False):
		finalData = pd.DataFrame([])
		if not df_twitter.empty:
			users = df_twitter.userID.tolist()
			unique_users = set(users)
			for user in tqdm(unique_users):
				userData = self.getUserTimelineData(user,test_mode=test_mode)
				finalData = finalData.append(userData)
			return finalData
		else:
			return None

	# getting all of the twitter data
	def getTwitterData(self, df, friendOpt = False,test_mode = False):
		if 'tweetID' in df:
			logging.info('[INFO] extraction started for twitter data')
			df_twitter = pd.DataFrame([])
			if (len(df) > 100):  # to limit the size for api to 100
				batchSize = int(math.ceil(len(df) / 100))
				for i in tqdm(range(batchSize)):
					logging.info("[INFO] batch %d started for Twitter data", i )
					dfBat = df[(100 * i):(100 * (i + 1))]
					temp = self.getBatchTwitter(dfBat.tweetID.tolist(),friendOpt = friendOpt,test_mode=test_mode)
					df_twitter = df_twitter.append(temp)
			else:
				logging.info("[INFO] single batch started for Twitter data")
				df_twitter = self.getBatchTwitter(df.tweetID.tolist(),friendOpt=friendOpt,test_mode=test_mode)
			# data.set_index('tweetId')
			return (df_twitter)

	def output(self, df, filename):
		os.chdir('../../input/')
		util.output_to_csv(df, filename=filename)
		logging.info("[INFO] CSV file created")

def main():
	parser = argparse.ArgumentParser(description='Extracting data from hexagon and twitter API')
	parser.add_argument('-o', '--friendOption', help='If friend list is required or not', default=False, type=util.str2bool)
	parser.add_argument('-f', '--filenameTwitter', help = 'specify the name of the file to be stored', default="hexagonData.csv")
	parser.add_argument('-f2', '--filenameFriends', help = 'specify the name of the file for following data(friends)', default="friendsData.csv" )
	parser.add_argument('-f3', '--filenameUserTimeline', help ='specify the name of the file for user timeline', default="userTimelineData.csv")
	parser.add_argument('-t', '--testMode', help = 'test modde to get only sample data, boolean True or False',type=util.str2bool,default=False)
	parser.add_argument('-u', '--userOption', help= 'If user timeline for each user needs to be extracted or not',default=False ,type=util.str2bool)
	args = vars(parser.parse_args())
	option = True if (args['friendOption'] == True) else False
	userOption = True if (args['userOption'] == True) else False
	logging.info('[NEW] ---------------------------------------------')
	logging.info('[INFO] new extraction process started ' + ('with friends option' if option == True else 'without the friends option'))
	test_mode = True if (args['testMode'] == True) else False
	ob = Hexagon(test_mode)
	df = ob.hexagonData
	filenameTwitter = args['filenameTwitter']
	filenameFriends = args['filenameFriends']
	filenameUserTimeline = args['filenameUserTimeline']
	tweet_data = ob.getTwitterData(df, friendOpt=option, test_mode=test_mode)
	ob.output(tweet_data, filenameTwitter)
	if option == True:
		logging.info("[INFO] extracting friends data")
		friendsData = ob.getFriendData(tweet_data, test_mode=True)
		ob.output(friendsData, filenameFriends)
	if userOption == True:
		logging.info("[INFO] user timeline extraction started..might take some time")
		userTimeline = ob.getuserTimeline(tweet_data,test_mode = test_mode)
		ob.output(userTimeline,filenameUserTimeline)
	logging.info("[INFO] job completed succesfully")
	# util.playSound()               # just for testing

if (__name__ == '__main__'):
	main()
	# logging.info('[NEW] ---------------------------------------------')
	# friendOpt = True
	# ob = Hexagon(True)
	# df = ob.hexagonData
	# tweet_data= ob.getTwitterData(df, friendOpt=True, test_mode=True)
	# filename = 'hexagonData.csv'
	# ob.output(tweet_data, filename)
	# ob.output(tweet_data, filename)
	# if (friendOpt == True):
	# 	friendsData = ob.getFriendData(tweet_data, test_mode=True)
	# 	ob.output(friendsData, 'friendsData.csv')


