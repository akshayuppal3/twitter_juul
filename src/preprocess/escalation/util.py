import os

import git
import pandas as pd

pd.set_option('display.max_colwidth', -1)
import warnings

warnings.filterwarnings('ignore')
import numpy as np
import re
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support


## Read the labelled files and the poly_user
def get_git_root(path):
	git_repo = git.Repo(path, search_parent_directories=True)
	git_root = git_repo.git.rev_parse("--show-toplevel")
	return git_root


top_dir = os.path.join(get_git_root(os.getcwd()))
input_dir = os.path.join(get_git_root(os.getcwd()), "input")
embeddings_dir = os.path.join(get_git_root(os.getcwd()), "input", "embeddings")
annotatted_dir = os.path.join(input_dir, "annotated_data")
classifier_dir = os.path.join(get_git_root(os.getcwd()), "models", "classifier")
model_dir = os.path.join(get_git_root(os.getcwd()), "models")
poly_dir = os.path.join(model_dir, "poly_users")


def get_length(s):
	a = list(s.split())
	return (len(a))


def get_window_size(df):
	tweet_count = df.groupby(by="userID")["tweetId"].count()
	tweet_count = tweet_count.reset_index()
	window = int(tweet_count.tweetId.quantile(0.95))
	return window


def get_max_length(df):
	## max_length
	lengths = df["tweetText"].progress_apply(get_length)
	max_len = int(lengths.quantile(0.95))
	return (max_len)


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


def clean_text(text):
	text = re.sub(r'(https?://\S+)', "", text)  ## remove url
	text = re.sub(r'(\@\w+)', "author", text)  ## remove @ mentions with author
	text = re.sub(r'(@)', "", text)  ## remove @ symbols
	text = re.sub(r'(author)', "", text)  ## remove author
	text = re.sub(r'(#)', "", text)  ## removing the hashtags signal
	text = re.sub(r'(RT )', "", text)  ## remove the retweet info as they dont convey any information
	text = re.sub(r'(^:)', "", text)
	text = text.rstrip
	text = text.lstrip
	return (text)


def get_f1(Y_true, Y_pred):
	f_scores = precision_recall_fscore_support(Y_true, Y_pred)[2]
	supports = precision_recall_fscore_support(Y_true, Y_pred)[3]
	f1_num = 0
	for f_s, sup in zip(f_scores, supports):
		f1_num += (f_s * sup)
	f1 = f1_num / np.sum(supports)
	return (f1)


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


def get_embedding_matrix(keras_tkzr, word2vec):
	vocab_size = len(keras_tkzr.word_index) + 1
	embedding_matrix = zeros((vocab_size, 100))
	for word, i in keras_tkzr.word_index.items():
		embedding_vector = word2vec.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	return (embedding_matrix, vocab_size)
