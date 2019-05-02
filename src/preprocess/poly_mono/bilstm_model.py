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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import util
import os
import pickle

class Bilstm:

	## @param  text: text to train the the bilstm model on
	## @param embeding_path: path for the embedding file
	def __init__(self,text,label,embedding_path):
		self.max_len = 60  ## average no of words in text (= tweets)
		self.epoch = 10
		self.validation_split = 0.25
		self.text = text
		self.y = label
		self.w2v = self.get_embedding(embedding_path)
		self.tokenizer = self.get_tokenizer(text)
		self.le = LabelEncoder()
		self.vocab_size = len(self.tokenizer.word_index) + 1
		self.embedding_matrix = self.get_embeding_matrix(embedding_path)
		self.model = self.get_bilstm_model()


	def get_tokenizer(self,text):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(text)
		return tokenizer

	def get_embedding(self,embedding_path):
		embedding_file = open(embedding_path, "r")
		print("getting the embeddings")
		w2v_path = os.path.join(util.modeldir,"embeddings","w2v.pkl")
		if os.path.exists(w2v_path):
			word2vec = pickle.load(open(w2v_path, "rb"))
			return word2vec
		else:
			if (embedding_file):
				word2vec = dict()
				split = embedding_file.read().splitlines()
				for line in split:
					key = line.split(' ', 1)[0]  # the first word is the key
					value = np.array([float(val) for val in line.split(' ')[1:]])
					word2vec[key] = value
				print("dumping the w2v file")
				util.pickle_file(word2vec,os.path.join(util.modeldir,"embeddings","w2v.pkl"))
				return (word2vec)
			else:
				print("invalid file path")

	# @param: filepath: filepath of embedding file
	# @ return: embedding matrix for the embedding layer
	def get_embeding_matrix(self,file_path):
		vocab_size = self.vocab_size
		tokenizer = self.tokenizer
		word2vec = self.w2v
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

	def get_encoded_data(self,data_):
		encoded_docs = self.tokenizer.texts_to_sequences(data_)
		data = pad_sequences(encoded_docs,maxlen=self.max_len,padding='post')
		return data

	def split_data(self):
		self.le.fit(self.y)
		print("output categories :", self.le.classes_)
		train_data, test_data, Y_train ,Y_test = train_test_split(self.text,self.y, test_size=0.20, random_state=6)
		X_train = self.get_encoded_data(train_data)
		X_test = self.get_encoded_data(test_data)
		Y_train = self.get_output_data(Y_train)
		Y_test = self.get_output_data(Y_test)
		return (X_train,X_test,np.array(Y_train), np.array(Y_test))

	def get_output_data(self,Y):
		y = self.le.transform(Y)
		return y
	
	def train(self,X_train,Y_train):
		model = self.model
		print("training the model")
		print("Y_train",Y_train.shape)
		print("X_train", X_train.shape)
		model.fit(X_train,y,validation_split=self.validation_split,nb_epoch=self.epoch,verbose=2)
		## printing the trainin scores
		print("predicting on training data")
		self.predict(X_train,y)
		self.model = model

	def predict(self,X_test, Y_test):
		model = self.model
		loss, accuracy = model.evaluate(X_test, Y_test, verbose=2)
		print('Accuracy: %f' % (accuracy * 100))
		Y_pred = model.predict(X_test)
		y_pred = np.array([np.argmax(pred) for pred in Y_pred])
		print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
