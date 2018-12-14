#############################
##Class for authentication###
##containing secrets key info
#############################
#############################
##TODO put all in doc string
import requests
class Authenticate:
	def __init__(self):
		self.consumer_key = 'EiLaKLlGvXdakLsKf7yi86pVo'
		self.consumer_secret = 'CWK0UzygY5sGD7lXEz9xOklaH7Fq6uaVyJ3ZZlEXXpYke1ImKD'
		self.access_token = "1069034330385715200-nq6kISbrjP3AcCBtvIbgepOxAeJmkF"
		self.access_secret = 'ahVZbT8Y8ydpRJGWzIkmSiIV0yiWsg5pB4ToMqOLhUjMk'
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
