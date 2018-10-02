#############################
###### Extracting data ######
## from hexagon API and then#
## passing to twitter API ###
############################
# @Author : Akshay

from authentication import Authenticate
from preprocessing import Preprocess
import requests
import urllib.request
import json
import pandas as pd
import util
import os
import math
import argparse
import logging
from logging.handlers import RotatingFileHandler
import tweepy

startDate = '2017-07-01'
endDate = '2018-10-01'
monitorID = "11553243040"  # juulMonitor twitter filter ID (numeric field)

logging.basicConfig(level="INFO", format= util.format, filename=(util.logdir + "/hexagonScrapingLogs.log"))
logger = logging.getLogger("logger")
# handler = RotatingFileHandler(util.logdir + "/hexagonScrapingLogs.log", maxBytes=10000000, backupCount=10)
# logger.addHandler(handler)

class Hexagon:
	def __init__(self):
		self.authenticateURL = "https://api.crimsonhexagon.com/api/authenticate"
		self.authToken = self.getAuthToken()
		self.dates = self.getDates()
		self.baseUrl = "https://api.crimsonhexagon.com/api/monitor"
		self.url = self.getURL()
		self.hexagonData = self.getData()

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
			print(e)

	def getDates(self):
		dates = "&start=" + startDate + "&end=" + endDate  # Combines start and end date into format needed for API call
		return dates

	def getEndPoint(self, endpoint):
		return '{}/{}?'.format(self.baseUrl, endpoint)

	def getURL(self):
		endpoint = self.getEndPoint('posts')
		extendLimit = "&extendLimit=true"  # extends call number from 500 to 10,000
		fullContents = "&fullContents=true"  # Brings back full contents for Blog and Tumblr posts which are usually truncated around sea
		url = '{}id={}{}{}{}{}'.format(endpoint, monitorID, self.authToken, self.dates, extendLimit, fullContents)
		return url

	def getJsonOb(self):
		webURL = urllib.request.urlopen(self.url)
		data = webURL.read().decode('utf8')
		theJSON = json.loads(data)
		return theJSON

	# @TODO check for more columns
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

	# @TODO check if no of posts < 10000
	def getData(self):
		logging.info('[INFO] extraction of Hexagon data started')
		JSON = self.getJsonOb()
		df = pd.DataFrame([])
		# if JSON['totalPostsAvailable'] <= 10000:
		for iter in JSON['posts']:
			data = self.getColumnsData(iter)
			df = df.append(data, ignore_index=True)
		logging.info('[INFO] all data extracted from hexagon')
		return df

	def getTwitterData(self, df, friends = False):
		if 'tweetID' in df:
			logging.info('[INFO] extraction started for twitter data')
			data = pd.DataFrame([])
			ob = Preprocess()
			api = ob.api
			if (len(df) > 100):  # to limit the size for api to 100
				batchSize = int(math.ceil(len(df) / 100))
				for i in range(batchSize):
					logging.info("[INFO] batch %d started for Twitter data", i )
					dfBat = df[(100 * i):(100 * (i + 1))]
					try:
						user = api.statuses_lookup(dfBat.tweetID.tolist())
						for idx, statusObj in enumerate(user):
							userData = ob.getTweetObject(tweetObj=statusObj, friendList=friends)
							data = data.append(userData, ignore_index=True)
					except tweepy.TweepError as e:
						logging.info("[Error] " + e.reason )
			else:
				try:
					user = api.statuses_lookup(df.tweetID.tolist())
					logging.info("[INFO] single batch  started for Twitter Data")
					for idx,statusObj in enumerate(user):
						userData = ob.getTweetObject(tweetObj=statusObj, friendList=friends)
						data = data.append(userData, ignore_index=True)
				except tweepy.TweepError as e:
					logging.info("[Error] " + e.reason)
			return data

	def output(self, df, filename):
		os.chdir('../output/hexagon')
		util.output_to_csv(df, filename=filename)
		logging.info("[INFO] CSV file created")

def main():
	parser = argparse.ArgumentParser(description='Extracting data from hexagon and twitter API')
	parser.add_argument('-f', '--friendList', help='If friend list is required or not', required=True)
	args = vars(parser.parse_args())
	logging.info("[INFO] new extraction process started")
	ob = Hexagon()
	df = ob.hexagonData
	if (args['friendList'] == True):
		option = True
	else:
		option = False
	df2 = ob.getTwitterData(df,friends=option)
	ob.output(df2, 'hexagonData2.csv')
	logging.info("[INFO] job completed succesfully")

if (__name__ == '__main__'):
	main()