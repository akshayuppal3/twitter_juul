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
from collections import deque
# import twintApi as tw

monitorID = "17244489159"  # juulMonitor twitter filter ID (numeric field)

logging.basicConfig(level="INFO", format= util.format, filename=(util.logdir + "/hexagonScrapingLogs.log"))
# logger = logging.getLogger("logger")
authenticateURL = "https://api.crimsonhexagon.com/api/authenticate"
baseUrl = "https://api.crimsonhexagon.com/api/monitor"

class Hexagon:
	def __init__(self,testMode = False,startDate = util.startDate,endDate = util.endDate):
		ob = Authenticate()
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
		start_dates = []
		end_dates = []
		today = dt_start
		while today < dt_end:
			tomorrow = today + one_day
			start_dates.append(today.strftime(util.dateFormat))
			end_dates.append(tomorrow.strftime(util.dateFormat))
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
		print(url)
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
	# @returns the hexagon data if data found else returns empty dataframe
	def getHexagonData(self, startD, endD, testMode = False):
		logging.info('[INFO] extraction of Hexagon data started')
		logging.info('[INFO] test mode selected') if testMode == True else None
		df = pd.DataFrame([])
		if (self.checkVolumeData(startD,endD)):
			print("Data being extracted in batches")
			logging.info('[INFO] Data being extracted in batches')
			startDates, endDates = self.getDateRange(startD,endD)
			for startD,endD in tqdm(zip(startDates,endDates), total= len(startDates)):
				data = self.getJSONData(startD,endD,testMode)
				df = df.append(data,ignore_index = True)
		else:
			df = self.getJSONData(startD,endD,testMode)
		logging.info('[INFO] all data extracted from hexagon')
		if (df.empty):
			print("No valid data found for the specified date range")
			logging.info('[INFO] no valid data found for the time range')
		if test_mode == True:
			df = df[0:util.testLimit] if (testMode == True) else df   # changes for the test mode
		print(len(df))
		return df

	def getBatchTwitter(self,api,tweetIDs, user_list,test_mode=False):
		data = pd.DataFrame([])
		try:
			user = api.statuses_lookup(tweetIDs,include_entities=True,tweet_mode='extended')  # update as per for full_text
			for idx, statusObj in enumerate(tqdm(user)):
				userData = util.getTweetObject(tweetObj=statusObj,user_list=user_list)
				if userData is not None:
					if not userData.empty:
						data = data.append(userData, ignore_index=True)
			return data

		except tweepy.TweepError as e:
			logging.error("[Error] " + e.reason)

	def getFriendBatch(self,friendIds,parent_id,test_mode = False):
		data = pd.DataFrame([])
		ob = Twitter()
		api_list = ob.api
		apis = deque(api_list)
		try:
			apis.rotate(-1)
			api = apis[0]
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
			for parentId,friendList in tqdm(zip(df_twitter.userID,df_twitter.friendList), total= len(df_twitter.userID)):
				if type(friendList) == str:
					friends = ast.literal_eval(friendList)  # as the friend list in the dataframe is string
				else:
					friends = friendList
				if len(friends) != 0:
					friends = friends[0:util.friendLimit]  # limiting the data size as the friends> 1000 are having errors
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
		else:
			return (pd.DataFrame([]))      #case for empty twitter data

	def getUserTimelineData(self,user,test_mode = False):
		ob = Twitter()
		userData = pd.DataFrame([])
		api_list = ob.api
		apis = deque(api_list)
		if user is not None:
			apis.rotate(-1)
			api = apis[0]
			# removing the count
			# count = util.testLimit if test_mode == True else util.userTimelineLimit
			try:
				status = self.api.user_timeline(user, tweet_mode = 'extended')
				for statusObj in status:
					data = ob.getTweetObject(statusObj,test_mode=test_mode)
					userData = userData.append(data)
				return userData

			except tweepy.TweepError as e:
				logging.error("[Error] " + e.reason)

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
	def getTwitterData(self, df_hex_tweets, user_list,filename,test_mode = False):
		api_list = self.api
		apis = deque(api_list)
		if filename.endswith('.csv'):
			filename, _ = filename.split('.csv')
		if 'tweetID' in df_hex_tweets:
			logging.info('[INFO] extraction started for twitter data')
			df_twitter = pd.DataFrame([])
			if (len(df_hex_tweets) > 100):  # to limit the size for api to 100
				batchSize = int(math.ceil(len(df_hex_tweets) / 100))
				for i in tqdm(range(batchSize)):
					apis.rotate(-1)
					api = apis[0]
					logging.info("[INFO] batch %d started for Twitter data", i )
					dfBat = df_hex_tweets[(100 * i):(100 * (i + 1))]
					temp = self.getBatchTwitter(api,dfBat.tweetID.tolist(),user_list,test_mode=test_mode)
					df_twitter = df_twitter.append(temp)
					if len(df_twitter) >= util.batch_file:
						file = str(str(filename) + '_' + str(i) + '.csv')
						file = os.path.join(util.inputdir,file)
						df_twitter.to_csv(file)
						df_twitter = pd.DataFrame()

			else:
				logging.info("[INFO] single batch started for Twitter data")
				df_twitter = self.getBatchTwitter(df_hex_tweets.tweetID.tolist(),test_mode=test_mode)
			# data.set_index('tweetId')
			return (df_twitter)
		else:
			return (pd.DataFrame())    # Case for a blank dataframe

	def getFriendsDataTwint(self,df):
		ob = tw.ScrapeTwitter()
		users = util.getUsers(df,type='ID')
		if users is not None:
			df_friends = ob.followingData(users)
			if not df_friends.empty:
				return df_friends
			else:
				return pd.DataFrame([])
		else:
			return pd.DataFrame([])

	def output(self, df, filepath):
		util.output_to_csv(df, filename=filepath)
		logging.info("[INFO] CSV file created")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting data from hexagon and twitter API')
	parser.add_argument('-o', '--friendOption', help='If friend list is required or not', default=False, type=util.str2bool)
	parser.add_argument('-f', '--filenameTwitter', help = 'specify the name of the output filename to be stored', default="hexagon_extract.csv")
	parser.add_argument('-f2', '--filenameFriends', help = 'specify the name of the file for following data(friends)', default="friendsData.csv" )
	parser.add_argument('-f3', '--filenameUserTimeline', help ='specify the name of the file for user timeline', default="userTimelineData.csv")
	parser.add_argument('-t', '--testMode', help = 'test modde to get only sample data, boolean True or False',type=util.str2bool,default=False)
	parser.add_argument('-u', '--userOption', help= 'If user timeline for each user needs to be extracted or not',default=False ,type=util.str2bool)
	parser.add_argument('-s', '--startDate', help = 'Specify the start date/since for extraction (yyyy-mm-dd)', default=util.startDate)
	parser.add_argument('-e', '--endDate', help = 'Specify the end date (yyyy-mm-dd)', default=datetime.datetime.today().strftime('%Y-%m-%d'))
	args = vars(parser.parse_args())
	friendOpt = True if (args['friendOption'] == True) else False
	userOption = True if (args['userOption'] == True) else False
	logging.info('[NEW] ---------------------------------------------')
	logging.info('[INFO] new extraction process started ' + ('with friends option' if friendOpt == True else 'without the friends option'))
	test_mode = True if (args['testMode'] == True) else False
	startDate = args['startDate']
	endDate = args['endDate']
	ob = Hexagon(test_mode,startDate,endDate)
	df_hex_tweets = ob.hexagonData
	output_filename = args['filenameTwitter']
	output_filepath = os.path.join(util.inputdir,output_filename)
	filenameFriends = args['filenameFriends']
	filenameUserTimeline = args['filenameUserTimeline']
	if (not df_hex_tweets.empty):
		# filter the data based on the userID
		user_path = os.path.join(util.inputdir,"users_list.xlsx")
		if (os.path.exists(user_path)):
			df_users = util.read_excel(user_path)
			users = util.getUsers(df_users,"ID")
			tweet_data = ob.getTwitterData(df_hex_tweets,users,output_filename,test_mode=test_mode)   # using twint for friends data
			ob.output(tweet_data, output_filepath)
			if friendOpt == True:
				logging.info("[INFO] extracting friends data")
				friendsData = ob.getFriendsDataTwint(tweet_data,)
				ob.output(friendsData, filenameFriends)
				# friendsData = ob.getFriendData(tweet_data, test_mode=True)
			if userOption == True:
				logging.info("[INFO] user timeline extraction started..might take some time")
				userTimeline = ob.getuserTimeline(tweet_data,test_mode = test_mode)
				ob.output(userTimeline,filenameUserTimeline)
	else:
		print("no hexagon data extarcted")
	logging.info("[INFO] job completed succesfully")
	print("Job completed")

# if (__name__ == '__main__'):
# 	main()
	# ob = Hexagon(False, '2018-09-05',)
	# df = ob.hexagonData
	# tweet_data = ob.getTwitterData(df,friendOpt= True)
	# ob.output(tweet_data,'hexagonDataNew.csv')

