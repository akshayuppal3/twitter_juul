# This function is related to pre-processing data
import pandas as pd
from tqdm import tqdm
import numpy as np
import util
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
tqdm.pandas()

## prepare data and splits the data into train and test
def prepare_data(input_data, users_labelled):
	## preapring the user data
	user_data = input_data.groupby(by="userID").agg({'tweetText': 'count',
	                                                 'followersCount': 'first',
	                                                 'friendsCount': 'first',
	                                                 'statusesCount': 'first',
	                                                 'favourites_count': 'first',
	                                                 'listedCount': 'first',
	                                                 }).reset_index()
	user_data = user_data.rename(columns={'tweetText': 'tweetCount'})
	
	# preapring text
	tweet_data = input_data.groupby(by="userID")["tweetText"].apply(lambda x: "%s" % ' '.join(x)).reset_index()
	## cleaning the text
	tweet_data["tweetText"] = tweet_data["tweetText"].progress_apply(util.clean_text)
	tweet_data["tweetText"] = tweet_data["tweetText"].progress_apply(util.get_tokens).str.join(" ")
	
	## merging the text and user data
	final_data = user_data.join(tweet_data.set_index("userID"), on="userID", how="inner").reset_index()
	final_data = final_data.fillna(0)
	
	## extract the labels
	y = list(final_data.join(users_labelled.set_index("userID"), on="userID", how="inner")["label"])
	print("oversampling")
	
	## downsampling based on userIDS
	userIDs = np.array(list(final_data.userID)).reshape(-1, 1)
	# rus = RandomUnderSampler(random_state=0)
	rus = RandomOverSampler(random_state=0)  ## oversampling instead of under
	rus.fit(userIDs, y)
	userIDs, y_sam = rus.fit_sample(userIDs, y)
	print("userIDS len",len(userIDs.flatten()))
	print(userIDs.flatten())
	input_data = (final_data.loc[final_data.userID.isin(userIDs.flatten())])
	print("oversampled data length", len(input_data))
	
	print("train-test split")
	train_data, test_data, Y_train, Y_test = train_test_split(input_data, y_sam, test_size=0.20, random_state=4,
	                                                          shuffle=True,
	                                                          stratify=y_sam)
	return (train_data, test_data, Y_train, Y_test)


## return user fatures
def prepare_user_features(input_):
	user_data = input_[["followersCount", "friendsCount", "statusesCount"
		, "favourites_count", "listedCount", "tweetCount", ]]
	## followerss/ friends ration
	user_data["ff_ratio"] = user_data["followersCount"] / user_data["friendsCount"]
	
	## using log of each of the columns
	user_data[["followersCount", "friendsCount"
		, "statusesCount", "favourites_count",
		       "listedCount"]] = np.log(
		user_data[["followersCount", "friendsCount", "statusesCount", "favourites_count",
		           "listedCount"]])
	
	user_data["unigrams"] = list(input_["tweetText"].apply(util.get_length))
	
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
	print("year", year)
	users_ = list(first_data["userID"].loc[
		              ((first_data.juul_first.dt.year <= year) & (first_data.juul_first.dt.year > (year - 1)))
		              & ((first_data.weed_first.dt.year == (year + 1)) | (pd.isnull(first_data.weed_first)))
		              ## weed data after 2015
		              ])  # users who will change after september
	
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
