def cal_text_pred(test_data, Y_test, model, tf_idf, svd):
	X_test = tf_idf.transform(test_data["tweetText"])
	X_test = svd.transform(X_test)  ## reduce the dimensionality
	y_pred = model.predict(X_test)
	return y_pred


def cal_user_pred(test_data, Y_test, model):
	X_test, _ = prepare_user_features(test_data)
	y_pred = model.predict(X_test)
	return y_pred


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
	
	user_data["unigrams"] = list(input_["tweetText"].apply(get_length))
	
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
