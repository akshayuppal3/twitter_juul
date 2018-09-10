import pandas as pd
import os
import tweepy
from tweepy import OAuthHandler
from authentication import Authenticate
import util

#constant
hashtags = '#juulvapor OR #juulnation OR #doit4juul OR #juul'
lang = "en"
inceptionDate = "2018-01-08"
## to remove keys from gitignore
class Preprocess:
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
        #change to accomodate rate limit errors
        api = tweepy.API(author)
        return(api)

    #@TODO check if more fields required or not
    # @params passing tweet, friendlist and userinfo(in case of following)
    # returns data frame of tweet and user info
    def getTweetObject(self, tweetObj, friendList, user=None):
        if user == None:
            data = pd.DataFrame(
                {
                    'tweetId': tweetObj.id_str,
                    'userID': tweetObj.user.id,
                    'parentID': 'None',
                    'favourites_count': tweetObj.user.favourites_count,
                    'userName': tweetObj.user.name,
                    'userDescription': tweetObj.user.description,
                    'userCreatedAt': tweetObj.user.created_at,
                    'imageurl': tweetObj.user.profile_image_url,
                    'userFollowersCount': tweetObj.user.followers_count,
                    'friendsCount': tweetObj.user.friends_count,
                    'friendList': [friendList]
                })
        else:
            data = pd.DataFrame(
                {
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
                }, index=[0])
        return data

    # function to get twitter and user info and return df
    # @params api: api_handler, hashtags: query parameters, inceptionDate: start date for hashtags, lang: for language restriction
    def getTwitterData(self, queryParams, inceptionDate, lang="en"):
        df = pd.DataFrame([])
        #     df.astype('object')       # as some data contains list
        try:
            for tweet in util.limit_handler(tweepy.Cursor(self.api.search, q=(queryParams), since_id=inceptionDate, lang=lang).items(5)):  # (lang=en) : lang restriction
                friendList = [friend for friend in
                              util.limit_handler(tweepy.Cursor(self.api.friends_ids, id=tweet.user.id).items(5))]  # items value set only for test
                data = self.getTweetObject(tweet, friendList=friendList)
                df = df.append(data, ignore_index=True, sort=False)
                for userID in friendList:
                    user = self.api.get_user(userID)
                    userData = self.getTweetObject(tweet, friendList=friendList, user=user)
                    if user.id not in df.userID:  # prevent duplication in data
                        df = df.append(userData, ignore_index=True, sort=False)
            return df
        except tweepy.TweepError as e:
            print(e.api_code)
            print(e.reason)

    def searchTweets(self):
        df = self.getTwitterData(hashtags, inceptionDate, lang)
        df = df.set_index('userID')  #Changing index for mongoDb
        os.chdir('../output')
        util.output_to_csv(df,filename='juulDataset.csv')


if(__name__ == '__main__'):
    ob = Preprocess()
    ob.searchTweets()
