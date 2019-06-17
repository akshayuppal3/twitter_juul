import os

import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer as keras_Tokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tqdm import tqdm
from xgboost import XGBClassifier
import lstm
import util

embeddings_dir = util.embeddings_dir
embedding_file = os.path.join(embeddings_dir, "glove.twitter.27B.100d.txt")

## @return data, X, y, embedding_matrix, max_len, vocalb
def prepare_data_lstm(df, users_labelled):
	dimension = 100
	
	df = df[["userID", "tweetText", "tweetId"]]
	## data
	print("length of the data", len(df))
	
	print("users", len(df.userID.unique()))
	
	## cleaning
	print("cleanining the data")
	tqdm.pandas()
	df["tweetText"] = df["tweetText"].progress_apply(util.clean_text)
	df["tweetText"] = df["tweetText"].progress_apply(util.get_tokens).str.join(" ")
	
	## print winodow , max_len for analysis purpose
	max_len = util.get_max_length(df)
	print("max_length", max_len)
	
	## undersampling based on the users_labelled file
	users = np.array(list(users_labelled["userID"])).reshape(-1, 1)
	label = list(users_labelled["label"])
	
	print("total users before", len(users))
	pos_samples = [ele for ele in label if ele == 1]
	print("length of positive samples before", len(pos_samples))
	rus = RandomUnderSampler(random_state=0)
	rus.fit(users, label)
	users_sample, label_samples = rus.fit_sample(users, label)
	users_sample = list(users_sample)  ## changing the shape
	
	## pos samples after
	pos_samples = [ele for ele in label_samples if ele == 1]
	print("length of positive samples after", len(pos_samples))
	print("total users after", len(users_sample))
	
	# adjusting the dataframe based on random undersampling
	print("before data", len(df))
	print("users before", len(df.userID.unique()))
	df = df.loc[df.userID.isin(users_sample)]  ## under sampled data
	print("after data", len(df))
	print("after users", len(df.userID.unique()))
	
	data = df.groupby(by="userID")["tweetText"].apply(lambda x: "%s" % ' '.join(x)).reset_index()
	data = data.join(users_labelled.set_index("userID"), on="userID", how="inner")
	
	## prepare the tokenizer
	print("preparing the tokenizer")
	keras_tkzr = keras_Tokenizer()
	keras_tkzr.fit_on_texts(data["tweetText"])
	vocab_size = len(keras_tkzr.word_index) + 1
	print("vocalb", vocab_size)
	
	## embedding matrix
	print("creating glove embeddign matrix")
	embedding_matrix = util.get_embedding_matrix(vocab_size, dimension, embedding_file,
	                                             keras_tkzr)  ## tokenizer contains the vocalb info
	
	## encoding the docs
	print("encoding the data")
	encoded_docs = keras_tkzr.texts_to_sequences(data["tweetText"])
	X = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
	
	y = np.array(list(data["label"]))
	
	print("X", X.shape)
	print("y", y.shape)
	
	return (data, X, y, embedding_matrix, max_len, vocab_size)

## run lstm model
def run_lstm_model(X, y, embedding_matrix, max_len, vocab_size, dimension, epoch, metrics=lstm.Metrics(), weights=None):
	## split the data
	print("train-test split")
	X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.20, random_state=4, shuffle=True, stratify=y)
	
	X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.25, random_state=4, shuffle=True,
	                                                  stratify=Y_train)
	
	print("X-train", X_train.shape)
	print("X-test", X_test.shape)
	
	print("creating lstm model")
	model = lstm.get_lstm_model(max_len, vocab_size, dimension, embedding_matrix)
	
	print("training the model with balance dataset")
	
	history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), nb_epoch=epoch,
	                    verbose=1, batch_size=32, class_weight=weights, callbacks=[metrics])
	
	##plotting trainin validation - no point as we dont want ot look at accuarcy
	lstm.training_plot(history)
	
	print("generating classfication report")
	loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
	print('Accuracy: %f' % (accuracy * 100))
	## lstm model
	temp = model.predict(X_test)
	y_pred = [np.argmax(value) for value in temp]
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
	f1_score = (util.get_f1(Y_test, y_pred))
	
	print("job finished")
	return (model, f1_score)

## @returns tfidf vectorized data
def prepare_data_tfidf(df, users_labelled):
	tqdm.pandas()
	df["tweetText"] = df["tweetText"].progress_apply(util.clean_text)
	df["tweetText"] = df["tweetText"].progress_apply(util.get_tokens).str.join(" ")
	data = df.groupby(by="userID")["tweetText"].apply(lambda x: "%s" % ' '.join(x)).reset_index()
	data = data.join(users_labelled.set_index("userID"), on="userID", how="inner")
	# ## prepare the tokenizer
	print("preparing the tokenizer")
	tf_idf = TfidfVectorizer(sublinear_tf=True)
	tf_idf.fit(data["tweetText"])
	X = tf_idf.fit_transform(data["tweetText"])
	y = np.array(list(data["label"]))
	
	print("downsampling")
	rus = RandomUnderSampler(random_state=0)
	rus.fit(X, y)
	X_sam, y_sam = rus.fit_sample(X, y)
	
	print("downsampled data length", (X_sam.shape))
	print("train-test split")
	
	X_train, X_test, Y_train, Y_test = train_test_split(X_sam, y_sam, test_size=0.20, random_state=4, shuffle=True,
	                                                    stratify=y_sam)
	return ((X_train, Y_train, X_test, Y_test), tf_idf)


## @ return scores from baseline models
def get_baseline_scores(X_train, Y_train, X_test, Y_test):
	print("training the models")
	print("svm")
	svm = LinearSVC(C=1, verbose=1)
	svm.fit(X_train, Y_train)
	
	print("rf")
	rf = RandomForestClassifier(n_estimators=100, max_depth=2,
	                            random_state=0)
	rf.fit(X_train, Y_train)
	
	print("xgBoost")
	xgb = XGBClassifier()
	xgb.fit(X_train, Y_train)
	
	print("predicting scores")
	print("svm")
	y_pred = svm.predict(X_test)
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
	svm_score = (util.get_f1(Y_test, y_pred))
	
	print("random_forest")
	y_pred = rf.predict(X_test)
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
	rf_score = (util.get_f1(Y_test, y_pred))
	
	print("xgboost")
	y_pred = xgb.predict(X_test)
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
	xgb_score = (util.get_f1(Y_test, y_pred))
	
	y_pred = [1 for x in range(len(Y_test))]
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
	maj_score = (util.get_f1(Y_test, y_pred))
	
	print("job finished")
	final = {
		'svm': [svm, svm_score],
		'rf': [rf, rf_score],
		'xg_boost': [xgb, xgb_score],
		'maj': [maj_score],
		# 'tf-idf': tf_idf,
	}
	return (final)

