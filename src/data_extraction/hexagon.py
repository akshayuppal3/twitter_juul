#############################
###### Extracting data ######
## from hexagon API and then#
## passing to twitter API ###
############################
# @Author : Akshay

from authentication import Authenticate
import requests
import urllib.request
import urllib.error
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
from dateutil.relativedelta import relativedelta
import time

monitorID = "9925794735"  # juulMonitor twitter filter ID (numeric field)

logging.basicConfig(level="INFO", format= util.format, filename=(util.logdir + "/hexagonScrapingLogs.log"))
authenticateURL = "https://api.crimsonhexagon.com/api/authenticate"
baseUrl = "https://api.crimsonhexagon.com/api/monitor"

class Hexagon:
	def __init__(self,startDate = util.startDate,endDate = util.endDate):
		ob = Authenticate()
		self.authenticateURL = authenticateURL
		self.authToken = self.getAuthToken()
		self.baseUrl = baseUrl
		self.api = ob.api
		self.hexagonData = self.getHexagonData(startDate, endDate,)

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
		dt_start = datetime.datetime.strptime(begin, util.date_format)
		dt_end =  datetime.datetime.strptime(end, util.date_format)
		one_day = datetime.timedelta(1)
		start_dates = []
		end_dates = []
		today = dt_start
		while today < dt_end:
			tomorrow = today + one_day
			start_dates.append(today.strftime(util.date_format))
			end_dates.append(tomorrow.strftime(util.date_format))
			today = tomorrow
		end_dates.append(dt_end.strftime(util.date_format))
		return (start_dates,end_dates)

	# !! Depcprecated
	# similar date but with Timestamp added to it (eg 2015-01-01T00:10:00)
	def getTimeRange(self,date):
		## splitting day by every hour
		start_dates = []
		end_dates = []
		my_date = datetime.datetime.strptime(date, util.date_format)
		for hour in range(23):
			time = datetime.datetime.combine(my_date,datetime.time(hour,0))
			start_dates.append(time.strftime(util.time_format))
			time =  datetime.datetime.combine(my_date,datetime.time(hour+1,0))
			end_dates.append(time.strftime(util.time_format))
		time = datetime.datetime.combine(my_date, datetime.time(23, 0))
		start_dates.append(time.strftime(util.time_format))
		time = datetime.datetime.combine(my_date, datetime.time(23, 59, 59))
		end_dates.append(time.strftime(util.time_format))
		return  (start_dates,end_dates)

	# returns start and end dates divided in year interval
	def get_year_range(self,start, end):
		start_dates = []
		end_dates = []
		start_d = datetime.datetime.strptime(start, util.date_format)
		end_d = datetime.datetime.strptime(end, util.date_format)
		if (end_d - start_d).days >= 365:                           # splitting interval by each year
			start_dates.append(start_d.strftime(util.date_format))
			delta = relativedelta(years=1)
			new_date = start_d + delta
			while new_date < end_d:
				end_dates.append(new_date.strftime(util.date_format))
				start_dates.append(new_date.strftime(util.date_format))
				new_date += delta
			end_dates.append(end_d.strftime(util.date_format))
			return (start_dates, end_dates)
		else:
			return ([start], [end])

	def getEndPoint(self, endpoint):
		return '{}/{}?'.format(self.baseUrl, endpoint)

	def getJsonOb(self,startD,endD):
		try:
			webURL = urllib.request.urlopen(self.getURL(startD,endD))
			data = webURL.read().decode('utf8')
			theJSON = json.loads(data)
			return theJSON
		except urllib.error.HTTPError as e:
			print("hhtp error raised: ",e.msg)
			logging.info("http error raised: %s"% e.msg)
			time.sleep(5)
			JSON = self.getJsonOb(startD,endD)
			print("resolved error: ", e.msg, " sleeping for 5")
			return JSON

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
			}, index=[0]
		)
		return data

	def getJSONData(self,startD,endD):
		JSON = self.getJsonOb(startD, endD)
		df = pd.DataFrame([])
		data = JSON['posts']
		for iter in tqdm(data):
			data = self.getColumnsData(iter)
			df = df.append(data, ignore_index=True)
		return df

	# return the totla volume of data (total Posts Available) by summing by each YEAR
	def checkVolumeData(self, startD, endD):
		## sum by year
		count = 0
		start_dates,end_dates = self.get_year_range(startD,endD)
		for start_d,end_d in zip(start_dates,end_dates):
			JSON = self.getJsonOb(start_d, end_d)
			if JSON:
				count += (JSON['totalPostsAvailable'])
		print("total valume of data",count)
		logging.info("total volume of data %d" % count)
		return count

	# works for data > 10000 (extract month wise data)
	# @returns the hexagon data if data found else returns empty dataframe
	def getHexagonData(self, startD, endD):
		logging.info('[INFO] extraction of Hexagon data started')
		df = pd.DataFrame([])
		if (self.checkVolumeData(startD,endD) > util.hexagon_limit):                      ## check if whole_data > 10k
			print("Data being extracted in batches")
			logging.info('[INFO] Data being extracted in batches')
			startDates, endDates = self.getDateRange(startD,endD)    ## splitting data by per day
			for startD,endD in tqdm(zip(startDates,endDates), total= len(startDates)):
				data = self.getJSONData(startD,endD)
				df = df.append(data,ignore_index = True)
		else:                                                        ## if whole_data < 10k
			df = self.getJSONData(startD,endD)
		logging.info('[INFO] all data extracted from hexagon')
		if (df.empty):
			print("No valid data found for the specified date range")
			logging.info('[INFO] no valid data found for the time range')
		print("length of the data",len(df))
		logging.info("length of hexagon extracted data %d" % len(df))
		return df

	# @param -> api, tweet Ids , user list
	# @return-> twiter data
	def getBatchTwitter(self,api,tweetIDs, user_list):
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

	# getting all of the twitter data
	# @ param : hex_tweets ->[List] (tweet IDs), filename ->[str], user list (list of users, default None)
	def getTwitterData(self, hex_tweets,filename,user_list=None):
		api_list = self.api
		apis = deque(api_list)
		if filename.endswith('.csv'):
			filename, _ = filename.split('.csv')
		if hex_tweets:
			logging.info('[INFO] extraction started for twitter data')
			df_twitter = pd.DataFrame([])
			if (len(hex_tweets) > 100):  # to limit the size for api to 100
				batchSize = int(math.ceil(len(hex_tweets) / 100))
				for i in tqdm(range(batchSize)):
					apis.rotate(-1)
					api = apis[0]
					logging.info("[INFO] batch %d started for Twitter data", i )
					dfBat = hex_tweets[(100 * i):(100 * (i + 1))]
					temp = self.getBatchTwitter(api,dfBat,user_list)
					df_twitter = df_twitter.append(temp)
					if len(df_twitter) >= util.batch_file:
						file = str(str(filename) + '_' + str(i) + '.csv')
						file = os.path.join(util.inputdir,file)
						df_twitter.to_csv(file)
						df_twitter = pd.DataFrame()

			else:
				apis.rotate(-1)
				api = apis[0]
				logging.info("[INFO] single batch started for Twitter data")
				df_twitter = self.getBatchTwitter(api,hex_tweets,user_list)
			return (df_twitter)
		else:
			return (pd.DataFrame())    # Case for a blank dataframe


	def output(self, df, filepath):
		util.output_to_csv(df, filename=filepath)
		logging.info("[INFO] CSV file created")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Extracting data from hexagon and twitter API')
	parser.add_argument('-u', '--user_filter',help='filter data for specific users', default=False, type=util.str2bool)
	parser.add_argument('-f', '--filename_twitter', help = 'specify the name of the output filename to be stored', default="hexagon_extract.csv")
	parser.add_argument('-s', '--start_date', help = 'Specify the start date/since for extraction (yyyy-mm-dd)', default=util.startDate)
	parser.add_argument('-e', '--end_date', help = 'Specify the end date (yyyy-mm-dd)', default=datetime.datetime.today().strftime('%Y-%m-%d'))
	args = vars(parser.parse_args())
	logging.info('[NEW] ---------------------------------------------')
	startDate = args['start_date']
	endDate = args['end_date']
	ob = Hexagon(startDate,endDate)
	df_hex_tweets = ob.hexagonData
	output_filename = args['filename_twitter']
	output_filepath = os.path.join(util.inputdir,output_filename)
	if (not df_hex_tweets.empty):
		# filter the data based on the userID
		if args['user_filter']:         # gather data for specific users only
			user_path = os.path.join(util.inputdir,"extraction","users_list.csv")
			if (os.path.exists(user_path)):
				df_users = util.readCSV(user_path)
				users = util.getUsers(df_users,"ID")
				tweetIDs = list(set(list(df_hex_tweets['tweetID'])))  ## selecting unique list of tweet Ids
				tweet_data = ob.getTwitterData(tweetIDs,output_filename,users)   # using unique tweetIds
				ob.output(tweet_data, output_filepath)
		else:
			tweetIDs = list(set(list(df_hex_tweets['tweetID'])))  ## selecting unique list of tweet Ids
			tweet_data = ob.getTwitterData(tweetIDs,output_filepath)
			ob.output(tweet_data, output_filepath)

	else:
		print("no hexagon data extarcted")
	logging.info("[INFO] job completed succesfully")
	print("Job completed")


