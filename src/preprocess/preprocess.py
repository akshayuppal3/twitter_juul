#############################
##Class for preprocessing####
#############################
#############################

from w2v import MeanEmbeddingVectorizer
import numpy as np
import pandas as pd
import util


class Preprocess:

	# @ param sentences,,w2v
	def __init__(self, w2v, df):
		self.w2v = w2v
		self.df = df  # one for extracting tweets

	def create_features(self):
		df = self.df
		df_new = pd.DataFrame([])
		if isinstance(df, pd.DataFrame):  # check if the object is dataframe
			if ('tweetText' in df.columns):
				df_new['no_urls'] = df['tweetText'].str.count(r'(https?://\S+)')
				df_new['no_authors'] = df['tweetText'].str.count(r'(\@\w+)')
				if ('hashtags' in df.columns):
					df_new['no_hashtags'] = df['hashtags'].apply(util.hashtag_count)
				# modifying the text after we got the features
				tweet_texts = pd.DataFrame(df['tweetText'].str.replace(r'(https?://\S+)', ""))  # urls
				tweet_texts = pd.DataFrame(tweet_texts['tweetText'].str.replace(r'(\@\w+)', "author"))  # author mentions
				tweet_texts = pd.DataFrame(tweet_texts['tweetText'].str.replace(r'(\#\w+)', ""))  # removing hashtags
				df_new['tweetText'] = tweet_texts
				df_new['no_words'] = df['tweetText'].apply(lambda x: len(x.split(' ')))
				return (df_new)
			else:
				print("invalid input dataframe")
		else:
			print("please specify a valid dataframe")

	# combination of meanembeddignvecotrizer and create_features
	# param sentences, w2v model, full dataframe with all columns
	def get_X(self, sentences, features='w2v'):
		w2v = self.w2v
		ob = MeanEmbeddingVectorizer(w2v)
		w2v_features = ob.transform(sentences)
		# getting other features
		if (features == 'both'):
			df_temp = self.create_features()
			tweet_features = df_temp.loc[:, df_temp.columns != 'tweetText']
			X = np.hstack((w2v_features, tweet_features))
			return X
		else:
			return (w2v_features)

	# cleaning the column to remove urls, authors and hashtags
	def clean_text(self, column):
		df = self.df
		if (column in df.columns):
			tweet_texts = pd.DataFrame(df.column.str.replace(r'(https?://\S+)', ""))  # remove urls
			tweet_texts = pd.DataFrame(
				tweet_texts.column.str.replace(r'(\@\w+)', "author"))  # replacing author mentions
			df = pd.DataFrame(tweet_texts.column.str.replace(r'(\#\w+)', ""))  # removing hashtags
			return df
