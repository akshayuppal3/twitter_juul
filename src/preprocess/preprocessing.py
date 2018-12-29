import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import util
from tqdm import tqdm
import ast

pd.set_option('display.max_colwidth', -1)
tqdm.pandas()
class Priors():

	def __init__(self):
		self.timeline = pd.read_csv(util.inputdir + 'userTimelineData.csv.zip',lineterminator='\n')
		# following file created from the notebook
		self.user_char = pd.read_csv(util.inputdir + 'user_characteristics.csv',index_col=0,keep_default_na=False) # manually_labelled
		self.X = self.user_char[self.user_char.columns[~self.user_char.columns.isin(['userID','userName','Promoter/not'])]]
		y = self.user_char['Promoter/not']
		self.y = y.replace('NA',np.nan)

	# used once and file created..
	def get_user_char(self):
		df_timeline = self.timeline
		# count no of hashtags
		df_timeline['hashtags'] = df_timeline['hashtags'].progress_apply(util.hashtag_count)
		df_timeline['urls_id'] = df_timeline['tweetText'].str.findall(r'(https?://\S+)') # find the occurence of urls
		df_timeline['urls'] = df_timeline['tweetText'].str.count(r'(https?://\S+)')  # count the no of urls per tweet
		df_timeline['author_mentions'] = df_timeline['tweetText'].str.count(r'(\@\w+)') # count no of user mentiosn per tweet
		df_timeline['tweetCreatedAt'] = pd.to_datetime(df_timeline.tweetCreatedAt)
		df_user = df_timeline.groupby(df_timeline['userID']).agg({'author_mentions': 'median',
		                                                          'urls': 'median',
		                                                          'retweetCount': 'sum'})
		total = df_user['retweetCount'].sum()
		df_user['retweetCount'] = df_user['retweetCount'].apply(lambda x: (x / total) * 100)
		df_tweets = df_timeline.groupby([df_timeline['userID'], df_timeline['tweetCreatedAt'].dt.date])['tweetId'].agg(
			'count')
		df_tweets = pd.DataFrame(df_tweets)
		df_user['min_tweets'] = df_tweets.reset_index(level=0).groupby(['userID']).min()
		df_user['max_tweets'] = df_tweets.reset_index(level=0).groupby(['userID']).max()
		df_user['avg_tweets'] = df_tweets.reset_index(level=0).groupby(['userID']).mean()
		df_tweets['diff'] = df_tweets.reset_index(level=0).index
		df_tweets['diff'] = abs(df_tweets['diff'] - df_tweets['diff'].shift())
		df_tweets['diff'] = df_tweets['diff'].apply(lambda x: x.days)
		# avg, min and max tweets
		# min , max , avg no of days in consecutive intervals..and total no of tweets
		df_user['min_interval'] = df_tweets.reset_index(level=0).groupby(['userID'])['diff'].min()
		df_user['max_interval'] = df_tweets.reset_index(level=0).groupby(['userID'])['diff'].max()
		df_user['avg_interval'] = df_tweets.reset_index(level=0).groupby(['userID'])['diff'].mean()
		df_user['total_tweets'] = df_tweets.groupby(['userID'])['tweetId'].sum()
		df_user['total_urls'] = df_timeline.groupby(['userID']).urls_id.sum().apply(lambda x: len(set(x)))
		return df_user

	def accuracy(self):
		X = self.X
		y = self.y
		X_sub = (X[~y.isin([np.nan])])
		y_sub = y[~y.isin([np.nan])]
		model = LogisticRegression(C=1.0)
		model = model.fit(X_sub, y_sub)
		y_pred = (model.predict(X_sub))
		score = accuracy_score(y_pred, y_sub)
		print(score)

if __name__ == '__main__':
	ob = Priors()
	# print(ob.timeline.columns)
	ob.accuracy()
