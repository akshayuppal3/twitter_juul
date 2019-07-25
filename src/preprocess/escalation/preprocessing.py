# This function is related to pre-processing data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import util

tqdm.pandas()


## prepare data and splits the data into train and test
def prepare_data(input_data, users_labelled):
	## preapring the user data
	user_data = input_data.groupby(by="userID").agg({'tweetText': 'count',
	                                                 'followersCount': 'first',
	                                                 'friendsCount': 'first',
	                                                 'favourites_count': 'first',
	                                                 'listedCount': 'first',
	                                                 }).reset_index()
	user_data = user_data.rename(columns={'tweetText': 'statusesCount'})
	# preapring text
	tweet_data = input_data.groupby(by="userID")["tweetText"].apply(lambda x: "%s" % ' '.join(x)).reset_index()
	## cleaning the text
	tweet_data["tweetText"] = tweet_data["tweetText"].progress_apply(util.clean_text)
	tweet_data["tweetText"] = tweet_data["tweetText"].progress_apply(util.get_tokens).str.join(" ")
	
	## merging the text and user data
	final_data = user_data.join(tweet_data.set_index("userID"), on="userID", how="inner").reset_index(drop=True)
	final_data = final_data.fillna(0)
	
	## extract the labels
	y = list(final_data.join(users_labelled.set_index("userID"), on="userID", how="inner")["label"])
	
	print("train-test split")
	train_data, test_data, Y_train, Y_test = train_test_split(final_data, y, test_size=0.20, random_state=4,
	                                                          shuffle=True,
	                                                          stratify=y)
	return (train_data, test_data, Y_train, Y_test)


## extracts user features from data
def prepare_user_features(input_):
	user_data = input_[["followersCount", "friendsCount", "statusesCount"
		, "favourites_count", "listedCount"]]
	## followerss/ friends ration
	user_data["ff_ratio"] = user_data["followersCount"] / user_data["friendsCount"]
	
	## using log of each of the columns
	user_data[["followersCount", "friendsCount"
		, "statusesCount", "favourites_count",
		       "listedCount"]] = np.log(
		user_data[["followersCount", "friendsCount", "statusesCount", "favourites_count",
		           "listedCount"]])
	
	user_data["unigrams"] = np.log(list(input_["tweetText"].apply(util.get_unique_length)))
	
	## replace the na and inf values
	user_data = user_data.replace([np.inf, -np.inf], np.nan)
	user_data.replace(np.nan, 0)
	user_data = user_data.fillna(0)
	
	## normalizing the values
	user_data = (user_data - user_data.min()) / (user_data.max() - user_data.min())
	user_data = user_data.replace([np.inf, -np.inf], np.nan)
	user_data = user_data.fillna(0)
	
	X = user_data.values
	return (X, user_data)


## @ returns the data in that year
def get_year_data(year, first_data, juul_data):
	data = juul_data[(juul_data.tweetCreatedAt.dt.year >= year - 1) & ((juul_data.tweetCreatedAt.dt.year) <= year)]
	all_users = data.userID.unique()  # all users in that year
	check_data = first_data[first_data.userID.isin(all_users)]  ## users we need to check
	selected_data = (check_data[
		(check_data.weed_first.dt.year > year) | (pd.isnull(check_data.weed_first))])  ## juul before
	selected_users = selected_data["userID"]  ## total juul before
	poly_users = selected_data["userID"][selected_data.weed_first.dt.year == (year + 1)]
	print("total_users", len(selected_users))
	print("users that will change", len(poly_users))
	users_lbl = pd.DataFrame(selected_users, columns=["userID"])
	users_lbl["label"] = 0
	users_lbl.loc[users_lbl.userID.isin(poly_users), "label"] = 1
	final_data = data[data.userID.isin(selected_users)]  ## filter data by juul before users
	print("total data", len(final_data))
	print("***********")
	return ((year, final_data, users_lbl))


## ! deprecate
## @ returns the data in that year
def get_year_data_old(year, first_data, juul_data):
	print("year", year)
	# Juul before users
	users_ = list(first_data["userID"].loc[
		              ((first_data.juul_first.dt.year <= year) & (first_data.juul_first.dt.year > (year - 1)))
		              & ((first_data.weed_first.dt.year == (year + 1)) | (pd.isnull(first_data.weed_first)))
		              ## weed data after 2015
		              ])  ## juul before users
	
	poly_turn = list(first_data["userID"].loc[
		                 (first_data.juul_first.dt.year <= year) &
		                 ((first_data.weed_first.dt.year == (year + 1)))])  ## for labelling based on the next year
	
	print("users that will change", len(poly_turn))
	print("total users", len(users_))
	
	## getting the input data
	data_ = juul_data.loc[juul_data.userID.isin(users_)]
	print("length of data", len(data_))
	
	## get label - they reamain same for this task as the no of users, we only change the tweets data
	users_ = data_.userID.unique()
	users_lbl = pd.DataFrame(users_, columns=["userID"])
	users_lbl["label"] = 0  ## initialize
	users_lbl.loc[users_lbl.userID.isin(poly_turn), "label"] = 1
	len(users_lbl.loc[users_lbl.label == 1])  ## sanity check
	return ((year, data_, users_lbl))


# get month data for between the interval start and end
def get_month_data(data_2018, first, end):
	bucket_ = data_2018.loc[
		(data_2018.tweetCreatedAt.dt.month >= first) & (data_2018.tweetCreatedAt.dt.month <= end)]  ## first
	print("length of the data", len(bucket_))
	print("total users", len(bucket_.userID.unique()))
	return bucket_
