## func related to lstm model
import matplotlib.pyplot as plt
import numpy as np
from keras import regularizers
from keras.callbacks import Callback
from keras.layers import (Dense, concatenate, Embedding, Bidirectional,
                          LSTM, Reshape)
from keras.models import Model, Input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer as keras_Tokenizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

import preprocessing


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
def create_model(max_len, vocalb_size, dimension, embedding_matrix):
	## handle text features..
	input = Input(shape=(max_len,))
	emb_word = Embedding(vocalb_size, dimension, weights=[embedding_matrix], input_length=max_len)(input)
	lstm_word = Bidirectional(LSTM(100, return_sequences=False, dropout=0.50, kernel_regularizer=regularizers.l2(0.01)),
	                          merge_mode='concat')(emb_word)
	
	# modelR = SpatialDropout1D(0.1)(modelR)
	output = Dense(2, activation='sigmoid')(lstm_word)
	model = Model(input, output)
	# sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
	print("compiling the model")
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
	return model


## handle two different inputs and then concatenate them (user and text features)
## return lstm mdoel with input = [words_in,user_in]
def create_model_comb(max_len, user_feature_len, vocalb_size, dimension, embedding_matrix):
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


def create_model_seq(max_len, window, vocab_size, dimension, embedding_matrix):
	max_len = max_len
	n_words = vocab_size
	Dimension = 100
	input = Input(shape=(window, max_len,))
	model = Embedding(n_words, dimension, weights=[embedding_matrix], input_length=(window, max_len,))(input)
	model = Reshape(target_shape=(window, (max_len * Dimension)))(model)
	# model =  Bidirectional (LSTM (100,return_sequences=True,dropout=0.25),merge_mode='concat')(model)
	model = Bidirectional(LSTM(100, return_sequences=False, dropout=0.25), merge_mode='concat')(model)
	output = Dense(2, activation='sigmoid')(model)
	model = Model(input, output)
	# sgd = SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
	print("compiling the model")
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	print(model.summary())
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


def get_encoded_data(data, keras_tkzr, max_len):
	## encoding the docs
	encoded_docs = keras_tkzr.texts_to_sequences(data)
	X = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
	return X


def fit_tokenizer(data):
	print("preparing the tokenizer")
	keras_tkzr = keras_Tokenizer()
	keras_tkzr.fit_on_texts(data)
	return keras_tkzr


## return cross_val mean score for each class
def get_cross_val_score(model, X, Y, n_splits, epoch=5):
	skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
	scores = []
	for fold, (train, test) in enumerate(skf.split(X, Y)):
		history = model.fit(X[train], Y[train], validation_split=0.25, nb_epoch=epoch,
		                    verbose=1, batch_size=32, class_weight=None, )
		training_plot(history)
		
		## prediction
		temp = model.predict(X[test])
		y_pred = [np.argmax(value) for value in temp]  ## sigmoid
		f1 = precision_recall_fscore_support(Y[test], y_pred, average=None)[2]
		print("fold =", fold)
		print('  Classification Report:\n', classification_report(Y[test], y_pred), '\n')
		scores.append(f1)
	score1 = np.mean([ele[0] for ele in scores])
	score2 = np.mean([ele[1] for ele in scores])
	
	print("*************")
	print(score1, score2)
	return (score1, score2)

