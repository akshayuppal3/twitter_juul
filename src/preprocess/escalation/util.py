#  helper functions

import os
import re

import git
import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.tokenize import TweetTokenizer

nltk.download('wordnet')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
tweet_tknzr = TweetTokenizer()


## Read the labelled files and the poly_user
def get_git_root(path):
	git_repo = git.Repo(path, search_parent_directories=True)
	git_root = git_repo.git.rev_parse("--show-toplevel")
	return git_root


## files
top_dir = os.path.join(get_git_root(os.getcwd()))
model_dir = os.path.join(get_git_root(os.getcwd()), "models")
input_dir = os.path.join(get_git_root(os.getcwd()), "input")
embeddings_dir = os.path.join(input_dir, "embeddings")
embedding_file = os.path.join(embeddings_dir, "glove.twitter.27B.100d.txt")


def get_tokens(sentence):
	#     tokens = nltk.word_tokenize(sentence)  # now using tweet tokenizer
	tokens = tweet_tknzr.tokenize(sentence)
	tokens = [token.lower() for token in tokens]
	tokens = [token for token in tokens if (token not in stopwords and len(token) > 1)]  ## remove punctuations
	tokens = [get_lemma(token) for token in tokens]
	return (tokens)


def get_lemma(word):
	lemma = wn.morphy(word)
	if lemma is None:
		return word
	else:
		return lemma


def get_max_length(df):
	## max_length
	lengths = df["tweetText"].progress_apply(get_length)
	max_len = int(lengths.quantile(0.95))
	return (max_len)


def get_length(s):
	a = list(s.split())
	return (len(a))


## get window size
def get_window_size(df):
	tweet_count = df.groupby(by="userID")["tweetId"].count()
	tweet_count = tweet_count.reset_index()
	window = int(tweet_count.tweetId.quantile(0.95))
	return window


def get_sequence(df, column, window, max_len):
	users = df.userID.unique()  # select the unique users
	X = []
	for user in tqdm(users):
		temp = list(df[column].loc[df.userID.isin([user])])
		if len(temp) < window:
			pad = np.zeros(((window - len(temp)), max_len))  # pad in case data is less than the window
			data = np.vstack((temp, pad))
		else:
			data = temp[:window]  ## truncate to be equal to window size
		X.append(data)
	return np.array(X)


## cleaning files
def clean_text(text):
	text = re.sub(r'(https?://\S+)', "", text)  ## remove url
	text = re.sub(r'(\@\w+)', "author", text)  ## remove @ mentions with author
	text = re.sub(r'(@)', "", text)  ## remove @ symbols
	text = re.sub(r'(author)', "", text)  ## remove author
	text = re.sub(r'(#)', "", text)  ## removing the hashtags signal
	text = re.sub(r'(RT )', "", text)  ## remove the retweet info as they dont convey any information
	text = re.sub(r'(^:)', "", text)
	text = text.rstrip()
	text = text.lstrip()
	return (text)


## returns the emnbedding matrix for the lstm model
def get_embedding_matrix(vocab_size, dimension, embedding_file, keras_tkzr):
	word2vec = get_word2vec(embedding_file)
	from numpy import zeros
	embedding_matrix = zeros((vocab_size, dimension))
	for word, i in keras_tkzr.word_index.items():
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	return embedding_matrix


# create the word2vec dict from the dictionary
def get_word2vec(file_path):
	file = open(file_path, "r")
	if (file):
		word2vec = dict()
		#         split = file.read().splitlines()
		for line in file:
			split_line = line.split(' ')
			key = split_line[0]  # the first word is the key
			value = np.array([float(val) for val in split_line[1:]])
			word2vec[key] = value
		return (word2vec)
	else:
		print("invalid fiel path")


def plot_coeff(k, model, feature_names):
	coef = (model.coef_.ravel())
	top_positive_coefficients = np.argsort(coef)[-k:]
	top_negative_coefficients = np.argsort(coef)[:k]
	top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
	# create plot
	plt.figure(figsize=(17, 8))
	colors = ['red' if c < 0 else 'blue' for c in coef[top_coefficients]]
	plt.bar(np.arange(2 * k), coef[top_coefficients], color=colors)
	feature_names = np.array(feature_names)
	plt.xticks(np.arange(1, 1 + 2 * k), feature_names[top_coefficients], rotation=60, ha='right', fontsize=20)
	plt.show()
	return coef
