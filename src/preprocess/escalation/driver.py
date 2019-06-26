# driver functions related to running user and text features

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from keras.preprocessing.text import Tokenizer as keras_Tokenizer
from keras.preprocessing.sequence import pad_sequences
import util
import preprocessing
import lstm
import numpy as np
import model


## @ return a trained svm model on text features for LR
def run_text_features(train_data, test_data, Y_train, Y_test):
	tf_idf = TfidfVectorizer(sublinear_tf=True)
	tf_idf.fit(train_data["tweetText"])  ## fit on train data
	
	## transform train and test data
	X_test = tf_idf.transform(test_data["tweetText"])
	X_train = tf_idf.transform(train_data["tweetText"])
	
	## reduce the dimesionality
	svd = TruncatedSVD(n_components=500, n_iter=7, random_state=42)
	svd.fit(X_train)
	X_train = svd.transform(X_train)
	X_test = svd.transform(X_test)
	
	scores, best_model = model.get_baseline_scores(X_train, X_test, Y_train, Y_test)
	return (scores, best_model[0], best_model[1], tf_idf, svd)

## pipeline for lstm model for processing user and text features.
## @return cross val scores, model, tokenizer and max_len
def run_lstm(train_data, test_data, Y_train, Y_test, dimension, epoch, weight=None):
	scores = []
	## print winodow , max_len for analysis purpose
	max_len = util.get_max_length(train_data)
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
	
	## getting the user features
	X_train_user, _ = preprocessing.prepare_user_features(train_data)
	X_test_user, _ = preprocessing.prepare_user_features(test_data)
	
	user_feat_len = (X_train_user.shape[1])
	print("creating lstm model")
	model = lstm.create_model(max_len, user_feat_len, vocab_size, dimension, embedding_matrix)
	
	print("training the model with balance dataset")
	history = model.fit([X_train, X_train_user], Y_train, validation_split=0.25, nb_epoch=epoch,
	                    verbose=1, batch_size=32, class_weight=weight, )
	
	##plotting trainin validation - no point as we dont want ot look at accuarcy
	lstm.training_plot(history)
	
	scores = lstm.get_cross_val_score(train_data, Y_train,dimension=dimension, n_splits=5, nb_epoch=epoch)
	
	print("generating classfication report")
	loss, accuracy = model.evaluate([X_test, X_test_user], Y_test, verbose=2)
	print('Accuracy: %f' % (accuracy * 100))
	## lstm model
	temp = model.predict([X_test, X_test_user])
	y_pred = [np.argmax(value) for value in temp]  ## sigmoid
	print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
	
	print("lstm cross val score ", np.array(scores).mean())
	
	print("job finished")
	return (scores, y_pred, model, keras_tkzr, max_len)

## run the pipeline for user features
def run_user_features(train_data, test_data, Y_train, Y_test):
	X_train, _ = preprocessing.prepare_user_features(train_data)
	X_test, _ = preprocessing.prepare_user_features(test_data)
	
	scores, best_model = model.get_baseline_scores(X_train, X_test, Y_train, Y_test)
	return (scores, best_model[0], best_model[1])

