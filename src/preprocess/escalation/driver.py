# driver functions containing running the whole pipeline related to running user and text features

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer as keras_Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_recall_fscore_support
import util
import preprocessing
import lstm
import numpy as np
import baselines


## run the pipeline for user features
def run_user_features(train_data, test_data, Y_train, Y_test,option="over"):
	X_train, _ = preprocessing.prepare_user_features(train_data)
	X_test, _ = preprocessing.prepare_user_features(test_data)
	print("before sampling postives in train ", util.get_postives(Y_train),"total lenggth:",len(Y_train))
	if (option == "over"):
		X_train, Y_train = util.get_oversample(X_train, Y_train)
		print("after sampling postives in train ", util.get_postives(Y_train),"total lenggth:",len(Y_train))
	elif (option == "under"):
		X_train, Y_train = util.get_undersample(X_train, Y_train)
		print("after sampling postives in train ", util.get_postives(Y_train),"total lenggth:",len(Y_train))
	all_models = baselines.get_baseline_scores(X_train, X_test, Y_train, Y_test)
	return (all_models)

## @ return a trained svm model on text features for LR
def run_text_features(train_data, test_data, Y_train, Y_test,option="over",svd=False):
	tf_idf = TfidfVectorizer(sublinear_tf=True)
	tf_idf.fit(train_data)  ## fit on train data
	
	## transform train and test data
	X_test = tf_idf.transform(test_data)
	X_train = tf_idf.transform(train_data)
	if (option =="over"):
		X_train, Y_train = util.get_oversample(X_train,Y_train)
	elif(option == "under"):
		X_train, Y_train = util.get_undersample(X_train, Y_train)
	
	## reduce the dimesionality
	if svd == True:
		svd = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
		svd.fit(X_train)
		X_train = svd.transform(X_train)
		X_test = svd.transform(X_test)
	
	baseline_models = baselines.get_baseline_scores(X_train, X_test, Y_train, Y_test)
	return (baseline_models, tf_idf, svd)

## pipeline for lstm model for processing user and text features.
## @return cross val scores, model, tokenizer and max_len
def run_lstm(train_data, test_data, Y_train, Y_test,
             dimension, epoch, cross_splits=5,option="over",cross_val= False,weight=None):
	scores = []
	## print winodow , max_len for analysis purpose
	max_len = util.get_max_length(train_data)
	if max_len > 60:
		max_len = 60
	print("max_length", max_len)
	
	## prepare the tokenizer
	keras_tkzr = lstm.fit_tokenizer(train_data)
	vocab_size = len(keras_tkzr.word_index) + 1
	print("vocalb", vocab_size)
	
	## embedding matrix
	print("creating glove embeddign matrix")
	embedding_matrix = util.get_embedding_matrix(vocab_size, dimension, util.embedding_file,
	                                        keras_tkzr)  ## tokenizer contains the vocalb info
	
	## encoding the docs
	print("encoding the data")
	X_train = lstm.get_encoded_data(train_data, keras_tkzr, max_len)
	
	X_test = lstm.get_encoded_data(test_data, keras_tkzr, max_len)
	
	print("X-train", X_train.shape)
	print("X-test", X_test.shape)
	
	## either of the options for under and over sampling
	if (option =="over"):
		X_train, Y_train = util.get_oversample(X_train,Y_train)
	elif(option == "under"):
		X_train, Y_train = util.get_undersample(X_train, Y_train)
	
	print("creating lstm model")
	model = lstm.create_model(max_len, vocab_size, dimension, embedding_matrix)
	
	
	print("training the model with balance dataset")
	history = model.fit(X_train, Y_train, validation_split=0.25, nb_epoch=epoch,
	                    verbose=1, batch_size=32, class_weight=weight, )
	
	##plotting trainin validation - no point as we dont want ot look at accuarcy
	lstm.training_plot(history)
	
	print("generating classfication report")
	loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
	print('Accuracy: %f' % (accuracy * 100))
	## lstm model
	temp = model.predict(X_test)
	y_pred = [np.argmax(value) for value in temp]  ## sigmoid
	print('  Classification Report of train data:\n', classification_report(Y_test, y_pred), '\n')
	
	if cross_val == True:
		print("first getting cross val scores")
		X = np.concatenate((X_train, X_test),axis=0)  ## for cross_val
		Y = np.array(list(Y_train) + list(Y_test))
		cross_scores = lstm.get_cross_val_score(model,X,Y,n_splits=cross_splits,epoch=epoch)
		print("lstm cross val scores ", (cross_scores))
	else:
		cross_scores = precision_recall_fscore_support(Y_test, y_pred, average=None)[2]
	
	print("job finished")
	return (cross_scores, y_pred, model, keras_tkzr, max_len)

# !! deprecated
def run_lstm_comb(train_data, test_data, Y_train, Y_test,
                  dimension, epoch, cross_splits=5,option="over",weight=None):
	scores = []
	## print winodow , max_len for analysis purpose
	max_len = util.get_max_length(train_data["tweetText"])
	if max_len > 60:
		max_len = 60
	print("max_length", max_len)
	
	## prepare the tokenizer
	print("preparing the tokenizer")
	keras_tkzr = keras_Tokenizer()
	keras_tkzr.fit_on_texts(train_data["tweetText"])
	vocab_size = len(keras_tkzr.word_index) + 1
	print("vocalb", vocab_size)
	
	## embedding matrix
	print("creating glove embeddign matrix")
	embedding_matrix = util.get_embedding_matrix(vocab_size, dimension, util.embedding_file,
	                                             keras_tkzr)  ## tokenizer contains the vocalb info
	
	## encoding the docs
	print("encoding the data")
	encoded_docs = keras_tkzr.texts_to_sequences(train_data["tweetText"])
	X_train = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
	
	## encoding the test data
	encoded_docs = keras_tkzr.texts_to_sequences(test_data["tweetText"])
	X_test = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
	
	print("X-train", X_train.shape)
	print("X-test", X_test.shape)
	
	## either of the options for under and over sampling
	if (option == "over"):
		X_train, Y_train = util.get_oversample(X_train, Y_train)
	elif (option == "under"):
		X_train, Y_train = util.get_undersample(X_train, Y_train)
	
	## getting the user features
	X_train_user, _ = preprocessing.prepare_user_features(train_data)
	X_test_user, _ = preprocessing.prepare_user_features(test_data)
	
	user_feat_len = (X_train_user.shape[1])
	print("creating lstm model")
	model = lstm.create_model_comb(max_len, user_feat_len, vocab_size, dimension, embedding_matrix)
	
	print("first getting cross val scores")
	X_text = np.concatenate((X_train, X_test),axis=0)  ## for cross_val
	X_user = np.concatenate((X_train_user,X_test_user),axis=0)
	X = [X_text,X_user]
	Y = np.array(list(Y_train) + list(Y_test))
	cross_scores = lstm.get_cross_val_score(model,X,Y,n_splits=cross_splits,epoch=epoch)
	
	print("training the model with balance dataset")
	history = model.fit([X_train, X_train_user], Y_train, validation_split=0.25, nb_epoch=epoch,
	                    verbose=1, batch_size=32, class_weight=weight, )
	
	##plotting trainin validation - no point as we dont want ot look at accuarcy
	lstm.training_plot(history)
	
	print("generating classfication report")
	loss, accuracy = model.evaluate([X_test, X_test_user], Y_test, verbose=2)
	print('Accuracy: %f' % (accuracy * 100))
	## lstm model
	temp = model.predict([X_test, X_test_user])
	y_pred = [np.argmax(value) for value in temp]  ## sigmoid
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
	
	print("lstm cross val score ", np.array(scores).mean())
	
	print("job finished")
	return (cross_scores, y_pred, model, keras_tkzr, max_len)
