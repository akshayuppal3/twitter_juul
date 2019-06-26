## func related to lstm model
import util
import numpy as np
import matplotlib.pyplot as plt
import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score

from keras.callbacks import Callback
from keras.models import Model, Input
from keras.layers import (Dense, concatenate,Flatten,SpatialDropout1D,Embedding,Bidirectional,
                          LSTM,TimeDistributed,Reshape,Average,Dropout)
from keras import regularizers
from keras.preprocessing.text import Tokenizer as keras_Tokenizer
from keras.preprocessing.sequence import pad_sequences

## plotting train and test plot
def training_plot(history):
	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.show()
	
	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

## calculated the lstm prediction
def cal_lstm_pred(test_data, Y_test, model, keras_tkzr, max_len):
	## encoding the test data
	encoded_docs = keras_tkzr.texts_to_sequences(test_data["tweetText"])
	X_test = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
	X_test_user, _ = preprocessing.prepare_user_features(test_data)
	## calculate the model predictions
	temp = model.predict([X_test, X_test_user])
	y_pred = [np.argmax(value) for value in temp]  ## sigmoid
	return y_pred


## handle two different inputs and then concatenate them (user and text features)
## return lstm mdoel with input = [words_in,user_in]
def create_model(max_len, user_feature_len, vocalb_size, dimension, embedding_matrix):
	## handle text features..
	words_in = Input(shape=(max_len,))
	emb_word = Embedding(vocalb_size, dimension, weights=[embedding_matrix], input_length=max_len)(words_in)
	lstm_word = Bidirectional(LSTM(100, return_sequences=False, dropout=0.50, kernel_regularizer=regularizers.l2(0.01)),
	                          merge_mode='concat')(emb_word)
	lstm_word = Dense(user_feature_len, activation='relu')(lstm_word)
	
	## takes the user features as input
	user_input = Input(shape=(user_feature_len,))
	
	## concatenate both of the features
	modelR = concatenate([lstm_word, user_input])
	# modelR = SpatialDropout1D(0.1)(modelR)
	output = Dense(2, activation='softmax')(modelR)
	model = Model([words_in, user_input], output)
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

## callback function for overriding epoch end etc.
class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	
	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
		val_predict = np.array([np.argmax(value) for value in val_predict])
		val_targ = self.validation_data[1]
		_val_f1 = f1_score(val_targ, val_predict)
		_val_recall = recall_score(val_targ, val_predict)
		_val_precision = precision_score(val_targ, val_predict)
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
		print("— val_f1: %f — val_precision: %f — val_recall %f" % (_val_f1, _val_precision, _val_recall))
		print('  Classification Report:\n', classification_report(val_targ, val_predict), '\n')
		return


## return cross_val mean score for each class
def get_cross_val_score(train_data, Y_train, dimension, n_splits, nb_epoch):
	scores = []
	train_ids = list(train_data.index)
	kFold = StratifiedKFold(n_splits=n_splits)
	for train, test in kFold.split(train_ids, Y_train):
		
		max_len = util.get_max_length(train_data.loc[train])
		if max_len > 60:
			max_len = 60
		print("max_length", max_len)
		
		## prepare the tokenizer
		print("preparing the tokenizer")
		keras_tkzr = keras_Tokenizer()
		keras_tkzr.fit_on_texts(train_data.loc[train]["tweetText"])
		vocab_size = len(keras_tkzr.word_index) + 1
		print("vocalb", vocab_size)
		
		## embedding matrix
		print("creating glove embeddign matrix")
		embedding_matrix = util.get_embedding_matrix(vocab_size, dimension, util.embedding_file,
		                                        keras_tkzr)  ## tokenizer contains the vocalb info
		
		X_train_user, _ = preprocessing.prepare_user_features(train_data.loc[train])
		X_test_user, _ = preprocessing.prepare_user_features(train_data.loc[test])
		
		encoded_docs = keras_tkzr.texts_to_sequences(train_data.loc[train]["tweetText"])
		X_train = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
		encoded_docs = keras_tkzr.texts_to_sequences(train_data.loc[test]["tweetText"])
		X_test = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
		
		user_feat_len = (X_train_user.shape[1])
		print("creating lstm model")
		model = create_model(max_len, user_feat_len, vocab_size, dimension, embedding_matrix)
		
		history = model.fit([X_train, X_train_user], Y_train[train], validation_split=0.25, nb_epoch=nb_epoch,
		                    verbose=1, batch_size=32, class_weight=None, )
		training_plot(history)
		
		## prediction
		temp = model.predict([X_test, X_test_user])
		y_pred = [np.argmax(value) for value in temp]  ## sigmoid
		f1 = precision_recall_fscore_support(Y_train[test], y_pred, average=None)[2]
		print(f1)
		print('  Classification Report:\n', classification_report(Y_train[test], y_pred), '\n')
		scores.append(f1)
	score1 = np.mean([ele[0] for ele in scores])
	score2 = np.mean([ele[1] for ele in scores])
	return (score1, score2)
