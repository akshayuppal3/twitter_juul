#############################
##Class for bilstm and    ##
## train and predict #######
#############################
#############################
from keras.preprocessing.text import Tokenizer
from numpy import zeros
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import ( Model, Input, Sequential)
from keras.layers import (  Dense, Flatten, Embedding, Bidirectional, LSTM , TimeDistributed, Average, Reshape)
from keras_contrib.layers import CRF
from sklearn.metrics import classification_report,confusion_matrix


class bilstm:

	## @param  text: text to train the the bilstm model on
	## @param embeding_path: path for the embedding file
	def __init__(self,text,embedding_path):
		self.tokenizer = self.get_tokenizer(text)
		self.vocab_size = len(self.tokenizer.word_index) + 1
		self.max_len = 60
		self.embedding_matrix = self.get_embeding_matrix(embedding_path)
		self.model = self.get_bilstm_model()
		self.epoch = 10
		self.validation_split = 0.25

	def get_tokenizer(self,text):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(text)
		return tokenizer

	def get_word2vec(self,file_path):
		file = open(file_path, "r")
		if (file):
			word2vec = dict()
			split = file.read().splitlines()
			for line in split:
				key = line.split(' ', 1)[0]  # the first word is the key
				value = np.array([float(val) for val in line.split(' ')[1:]])
				word2vec[key] = value
			return (word2vec)
		else:
			print("invalid file path")

	# @param: filepath: filepath of embedding file
	# @ return: embedding matrix for the embedding layer
	def get_embeding_matrix(self,file_path):
		vocab_size = self.vocab_size
		tokenizer = self.tokenizer
		word2vec = self.get_word2vec(file_path)
		embedding_matrix = zeros((vocab_size, 100))
		for word, i in tokenizer.word_index.items():
			embedding_vector = word2vec.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
		return embedding_matrix

	# @param get the model
	def get_bilstm_model(self):
		## pass to the bi-lstm model
		max_len = self.max_len
		vocab_size = self.vocab_size
		embedding_matrix = self.embedding_matrix
		input = Input(shape=(max_len,))
		model = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_len)(input)
		model = Bidirectional(LSTM(100, return_sequences=True, dropout=0.50), merge_mode='concat')(model)
		model = TimeDistributed(Dense(100, activation='relu'))(model)
		model = Flatten()(model)
		model = Dense(100, activation='relu')(model)
		output = Dense(3, activation='softmax')(model)
		model = Model(input, output)
		model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
		print(model.summary())
		return model

	def train(self,X_train,Y_train):
		model = self.model
		print("training the model")
		model.fit(X_train,Y_train,validation_split=self.validation_split,nb_epoch=self.epoch,verbose=2)
		self.model = model

	def predict(self,X_test, Y_test):
		model = self.model
		loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
		print('Accuracy: %f' % (accuracy * 100))
		Y_pred = model.predict(X_test)
		y_pred = np.array([np.argmax(pred) for pred in Y_pred])
		print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
