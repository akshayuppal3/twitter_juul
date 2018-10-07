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

startDate = '2017-05-01'
endDate = '2018-05-02'
monitorID = "11553243040"  # juulMonitor twitter filter ID (numeric field)
testLimit = 5

logging.basicConfig(level="INFO", format= util.format, filename=(util.logdir + "/hexagonScrapingLogs.log"))
# logger = logging.getLogger("logger")
authenticateURL = "https://api.crimsonhexagon.com/api/authenticate"
baseUrl = "https://api.crimsonhexagon.com/api/monitor"

class Hexagon:
	def __init__(self,testMode = False):
		self.authenticateURL = authenticateURL
		self.authToken = self.getAuthToken()
		self.baseUrl = baseUrl
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

	def getJSONData(self,startD,endD):
		JSON = self.getJsonOb(startD, endD)
		df = pd.DataFrame([])
		for iter in JSON['posts']:
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
			for startD,endD in zip(startDates,endDates):
				data = self.getJSONData(startD,endD)
				df.append(data,ignore_index = True)
		else:
			df = self.getJSONData(startD,endD)
		logging.info('[INFO] all data extracted from hexagon')
		return df[0:testLimit] if (testMode == True) else df   # changes for the test mode

	def getBatchTwitter(self,tweetIDs,friendOpt = False):
		data = pd.DataFrame([])
		ob = Twitter()
		api = ob.api
		try:
			user = api.statuses_lookup(tweetIDs,include_entities=True,tweet_mode='extended')  # update as per for full_text
			for idx, statusObj in enumerate(user):
				userData = ob.getTweetObject(tweetObj=statusObj, friendOpt=friendOpt)
				data = data.append(userData, ignore_index=True)
			return data
		except tweepy.TweepError as e:
			logging.error("[Error] " + e.reason)

	def getTwitterData(self, df, friendOpt = False):
		if 'tweetID' in df:
			logging.info('[INFO] extraction started for twitter data')
			data = pd.DataFrame([])
			if (len(df) > 100):  # to limit the size for api to 100
				batchSize = int(math.ceil(len(df) / 100))
				for i in range(batchSize):
					logging.info("[INFO] batch %d started for Twitter data", i )
					dfBat = df[(100 * i):(100 * (i + 1))]
					temp = self.getBatchTwitter(dfBat.tweetID.tolist(),friendOpt = friendOpt)
					data = data.append(temp)
			else:
				logging.info("[INFO] single batch started for Twitter data")
				data = self.getBatchTwitter(df.tweetID.tolist(),friendOpt=friendOpt)
			data.set_index('tweetId')
			return data

	def output(self, df, filename):
		os.chdir('../input/')
		util.output_to_csv(df, filename=filename)
		logging.info("[INFO] CSV file created")

def main():
	parser = argparse.ArgumentParser(description='Extracting data from hexagon and twitter API')
	parser.add_argument('-o', '--friendOption', help='If friend list is required or not', default=False, type=util.str2bool)
	parser.add_argument('-f', '--filename', help = 'specify the name of the file to be stored', default="hexagonDataset.csv")
	parser.add_argument('-t', '--testMode', help = 'test modde to get only sample data, boolean True or False',type=util.str2bool,default=False)
	args = vars(parser.parse_args())
	option = True if (args['friendOption'] == True) else False
	logging.info('[NEW] ---------------------------------------------')
	logging.info('[INFO] new extraction process started ' + ('with friends option' if option == True else 'without the friends option'))
	test_mode = True if (args['testMode'] == True) else False
	ob = Hexagon(test_mode)
	df = ob.hexagonData
	filename = args['filename']
	df2 = ob.getTwitterData(df,friendOpt=option)
	ob.output(df2, filename)
	logging.info("[INFO] job completed succesfully")
	util.playSound()               # just for testing

if (__name__ == '__main__'):
	main()