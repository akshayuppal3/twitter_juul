#############################
##Class for authentication###
##containing secrets key info
#############################
#############################
##TODO put all in doc string
import requests
class Authenticate:
	def __init__(self):
		self.consumer_key = 'bV031fLEksoIvV6gO3juVe5fw'
		self.consumer_secret = '3xjNTYfKvpgkkWRbi4blyC8GGVKuBjUPMGLXpab0w9IFiJ6ITv'
		self.access_token = "141309128-bDGWx9p1g6OS0uJZm945YVMetRKg9o6T9U2xv8uf"
		self.access_secret = 'SdP13GF6LZxbmB6bqkQpiKUdaZROvQxQDD9S8Kgb0lwHw'
		# crimson credentials
		self.username = "auppal8@uic.edu"
		self.password = "Xperia123@"

	def getConsumerKey(self):
		return(self.consumer_key)

	def getConsumerSecret(self):
		return(self.consumer_secret)

	def getAccessToken(self):
		return(self.access_token)

	def getAccessSecret(self):
		return(self.access_secret)
