from authentication import Authenticate
import requests
import urllib.request
import json
import pandas as pd
import util
import os

startDate = '2017-07-01'
endDate = '2018-09-25'
monitorID = "11553243040"      #juulMonitor twitter filter ID (numeric field)


class Hexagon:
	def __init__(self):
		self.authenticateURL = "https://api.crimsonhexagon.com/api/authenticate"
		self.authToken = self.getAuthToken()
		self.dates = self.getDates()
		self.baseUrl = "https://api.crimsonhexagon.com/api/monitor"
		self.url = self.getURL()

	def getAuthToken(self):
		ob = Authenticate()
		username = ob.username
		password = ob.password
		querystring = {
			"username": username,
			"noExpiration": "true",
			"password": password
		}
		try :
			response = requests.request("GET", headers={}, url=self.authenticateURL, params=querystring)
			if (response.status_code == 200):
				result = response.json()
				authToken = result['auth']
				authToken = "&auth=" + authToken
				return (authToken)
		except requests.ConnectionError as e:
			print(e)

	def getDates(self):
		dates = "&start="+startDate+"&end="+endDate #Combines start and end date into format needed for API call
		return dates

	def getEndPoint(self,endpoint):
		return '{}/{}?'.format(self.baseUrl,endpoint)

	def getURL(self):
		endpoint = self.getEndPoint('posts')
		extendLimit = "&extendLimit=true"  # extends call number from 500 to 10,000
		fullContents = "&fullContents=true"  # Brings back full contents for Blog and Tumblr posts which are usually truncated around sea
		url = '{}id={}{}{}{}{}'.format(endpoint,monitorID,self.authToken,self.dates,extendLimit,fullContents)
		return url

	def getJsonOb(self):
		webURL = urllib.request.urlopen(self.url)
		data = webURL.read().decode('utf8')
		theJSON = json.loads(data)
		return theJSON

	#@TODO check for more columns
	def getColumnsData(self,hexagonObj):
		data = pd.DataFrame(
			{
				'tweetID': hexagonObj['url'].split("status/")[1],
				'url' : hexagonObj['url'],
				'type' : hexagonObj['type'],
				'title': hexagonObj['title'],
				'location' : hexagonObj['location'] if 'location' in hexagonObj else "",
				'language' : hexagonObj['language']
			}, index=[0]
		)
		return data

	#@TODO check if no of posts < 10000
	def getData(self):
		JSON = self.getJsonOb()
		df = pd.DataFrame([])
		# if JSON['totalPostsAvailable'] <= 10000:
		for iter in JSON['posts']:
			data = self.getColumnsData(iter)
			df = df.append(data,ignore_index=True)
		return df

	def output(self):
		df = self.getData()
		# if (df):
		os.chdir('../output')
		util.output_to_csv(df,filename='hexagonData.csv')

def main():
	ob = Hexagon()
	ob.output()


if __name__ == '__main__':
	main()

