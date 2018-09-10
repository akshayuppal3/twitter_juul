#############################
##Class for authentication###
##containing secrets key info
#############################
#############################
##TODO put all in doc string
class Authenticate:
    def __init__(self):
        self.consumer_key = '#'                            
        self.consumer_secret = '#'
        self.access_token = "#"
        self.access_secret = '#'

    def getConsumerKey(self):
        return(self.consumer_key)

    def getConsumerSecret(self):
        return(self.consumer_secret)

    def getAccessToken(self):
        return(self.access_token)

    def getAccessSecret(self):
        return(self.access_secret)