#############################
##Class for creating#########
##w2v embeddings class#######
#############################
#############################
from gensim.models import Word2Vec
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle


class W2v:
	def __init__(self, sentences):
		self.sentences = sentences
		self.embeddings = self.get_w2v_vectors()

	def get_w2v_vectors(self):
		model = Word2Vec(self.sentences, size=20, min_count=2)
		w2v = dict(zip(model.wv.index2word, model.wv.syn0))
		return w2v

# dump the w2v in main

# class that returns word2vec
class MeanEmbeddingVectorizer(object):
	def __init__(self, word2vec):
		self.word2vec = word2vec
		# if a text is empty we should return a vector of zeros
		# with the same dimensionality as all the other vectors
		self.dim = len(next(iter(word2vec.values())))

	def fit(self, X, y):
		return self

	def transform(self, X):
		return np.array([
			np.mean([self.word2vec[w] for w in words if w in self.word2vec]
					or [np.zeros(self.dim)], axis=0)
			for words in tqdm(X)
		])
