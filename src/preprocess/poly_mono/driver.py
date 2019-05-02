#############################
##Class for predicting mono##
##and poly users and getting#
##baselines metrics   #######
#############################
#############################

import pickle
import warnings
import os
import argparse
import pandas as pd
from gensim.models import Word2Vec
from preprocess import Preprocess
from sklearn import preprocessing
from training import training
from bilstm_model import Bilstm
import util
from tqdm import tqdm
import numpy as np
# Suppress warning

embedding_path = os.path.join(util.embeddir,"glove.twitter.27B.100d.txt")
model_dir = util.modeldir
model_path = os.path.join(model_dir,"classifier_poly_mono","bilstm.json")


def warn(*args, **kwargs):
	pass

class Classify:


	# df_train_lbl is the 500 lablled data
	# df_timeline_lbl is the labelleing of teh entire input file

	def w2v_calc(self,df_input):
		sentences_input = util.get_sentences(df_input, 'tweetText')
		model = Word2Vec(sentences_input, size=100, min_count=2)  ## traing our corpus for extracting word2vec- <100-D>
		w2v = dict(zip(model.wv.index2word, model.wv.syn0))
		# dump the w2v embeddings
		w2v_file = os.path.join(util.modeldir,"w2v","w2v.pkl")
		with open(w2v_file, "wb") as f:
			pickle.dump(w2v, f)
		print("dumping the w2v model finished")

	def train(self,w2v,df_lbl):
		print("\n************")
		print(len(df_lbl))  ## delete
		print(type(df_lbl))
		sentences = util.get_sentences(df_lbl, 'tweetText')
		pre = Preprocess(w2v, df_lbl)

		## getting the features of the labelled data
		print("getting the features of labelled data")
		X = pre.get_X(sentences)
		y = list(df_lbl.label)
		# normalize the features
		X = preprocessing.normalize(X, norm='l1')
		y = [int(i) for i in y]
		train = training(X, y)
		best_model = train.train_baseline()  # it returns a tuple (model,name)

		print("training the bilstm model")
		lstm = Bilstm(sentences,y,embedding_path)
		self.tokenizer = lstm.tokenizer
		self.max_length = lstm.max_len

		X_train,X_test,Y_train,Y_test = lstm.split_data()
		lstm.train(X_train,Y_train)

		# model = best_model[0]
		# name = best_model[1]

		print("metrics for the bilstm model (test set)")
		lstm.predict(X_test,Y_test)

		## dump the bilstm model
		print("dumping the bilstm model")
		util.dump_model(lstm,model_path)

		# train_model_path = os.path.join(util.modeldir, 'train_model.pkl')
		# with open(train_model_path, "wb") as f:
		# 	pickle.dump(model, f)
		# print("taking the " + str(name) + " classifier to predict on the input")

	def predict(self,df_input):
		print("\n************")
		print(len(df_input))  ## delete
		pre = Preprocess(None, df_input)
		df_input = pre.clean_text('tweetText')

		print("predicting with bilstm model")
		## loading the bilstm model
		bilsmt_model =  util.load_model(model_path)# load json and create model

		# train = training(X)
		# y_pred = train.predict(train_model)

		X = util.get_encoded_data(list(df_input['tweetText']),self.tokenizer,self.max_length)
		Y_pred = bilsmt_model.predict(X)
		y_pred = np.array([np.argmax(pred) for pred in Y_pred])

		## get the no of labelled as 1, 2 and 3
		print("no of labelled as 3", len(y_pred[y_pred == 3]))
		print("no of labelled as 2", len(y_pred[y_pred == 2]))
		print("no of labelled as 1", len(y_pred[y_pred == 1]))
		df_input['label'] = y_pred
		return df_input

	def poly_mono(self,df_input_pred):
		weed_words = pickle.load(open(os.path.join(util.modeldir, "weed_words.pkl"), "rb"))
		weed_words = [(" " + word + " ") for word in weed_words]
		pattern_weed = "|".join(weed_words)
		pattern_juul = 'juul'
		print("extracting the pattern for juul")
		df_tweet_weeds = df_input_pred[df_input_pred['tweetText'].str.contains(pattern_weed, case=False)]
		index = df_tweet_weeds.index  # the file after filtering have the index contained within
		df_weeds = df_input_pred.loc[index]
		poly_users = list(set(list(df_weeds.loc[df_weeds.label == 3]['userID'])))
		total_users = list(df_input_pred.userID.unique())
		poly_length = len(poly_users)
		total_users_length = len(total_users)
		mono_length = total_users_length - poly_length
		print("total users = ", total_users_length)
		print("no of poly users = ", poly_length)
		print("no of mono users = ", mono_length)
		print("% of poly users is ", poly_length / total_users_length)
		print("% of mono users is ", mono_length / total_users_length)
		print("*** starting with the poly sub type users")
		poly_user1 = list()
		poly_user2 = list()
		poly_user3 = list()
		poly_und = list()
		total_users = list(df_input_pred.userID.unique())
		for user in tqdm(poly_users):
			user_tweets = df_input_pred.loc[df_input_pred.userID == user]
			user_tweets.sort_values(by='tweetCreatedAt', ascending=True,
			                        inplace=True)  # sort by tweet created at
			juul_tweets = user_tweets[user_tweets['tweetText'].str.contains(pattern_juul, case=False)]
			juul_tweets.reset_index(drop=True, inplace=True)
			if (len(juul_tweets) > 0):
				time_j = pd.to_datetime(
					juul_tweets.head(1)['tweetCreatedAt'].values[0])  # getting the tweet with
			else:
				time_j = None
			weed_tweets = user_tweets[user_tweets['tweetText'].str.contains(pattern_weed, case=False)]
			if (len(weed_tweets) > 0):
				weed_tweets_user = weed_tweets[weed_tweets.label == 3]
				if (len(weed_tweets_user) > 0):
					times_w = pd.to_datetime(list(weed_tweets_user[
						                              'tweetCreatedAt']))
			else:
				times_w = None
			if (time_j != None and times_w is not None):
				pos = list(times_w).index(util.nearest((times_w), time_j))
				if (pos >= len(times_w)):
					poly_user1.append(user)
				elif (pos == 0):
					poly_user2.append(user)
				else:
					poly_user3.append(user)
			else:
				poly_und.append(user)
		print("Poly type users calculated")
		print("total users =", len(total_users))
		print("****************\n")
		print("% of pol1 users = ", len(poly_user1) / len(poly_users))
		print("\n")
		print("% of pol2 users = ", len(poly_user2) / len(poly_users))
		print("\n")
		print("% of pol3 users = ", len(poly_user3) / len(poly_users))
		print("\n")
		print("% of undefined users = ", len(poly_und) / len(poly_users))

def main():
	warnings.warn = warn
	parser = argparse.ArgumentParser(description="getting the poly and the mono users")
	parser.add_argument("-a",'--annotated_file',help="specify the annotated Dataframe containing labelled file",required=True)
	parser.add_argument("-f",'--function',help="specify the option as ( w2v / train / predict/ users)", default='train')  # option for train or prediction
	parser.add_argument("-p","--pred_file",help="specify the file to get prediction on ",required=False)
	args = vars(parser.parse_args())

	clasify = Classify()
	# creating embeddings from the input file
	if args['function'] == 'w2v':
		if (args['annotated_file']):
			filename = args['annotated_file']
			df_input = util.read_file(filename)
			print("creating w2v might take some time (5-10min)")
			clasify.w2v_calc(df_input)
		else:
			print("specify the input file")

	# either using the pickled annotated file or passing the annotated file
	# training the model
	elif args['function'] == 'train':
		# user the labelled file to get classification accuracy with different models
		w2v_filename = os.path.join(util.modeldir,"w2v","w2v.pkl")
		if (os.path.exists(w2v_filename)):  ## check if embedding exists
			w2v = pickle.load(open(w2v_filename, "rb"))
			if (w2v):
				if args['annotated_file']:
					filename = args['annotated_file']
					lbl_file = pd.read_csv(filename)
					clasify.train(w2v,lbl_file)
				else:
					if (os.path.exists(os.path.join(util.modeldir + "df_train_lb.pkl"))):
						lbl_file_path = os.path.join(util.modeldir + "df_train_lb.pkl")
						print("loading the labelled file")
						df_lbl = pickle.load(open(lbl_file_path,"rb"))
						clasify.train(w2v,df_lbl)
					else:
						print("labelled file does not exist")
			else:
				print("w2v embeddings file does not exist, please user -f <w2v> option")

	# predicting the model
	elif args['function'] == 'predict':
		if (args['pred_file']):
			input_file_path = args['pred_file']
			df_input = util.read_file(input_file_path)
			df_input_pred = clasify.predict(df_input)
			clasify.poly_mono(df_input_pred)
			# dump the final full labelled dataset
			df_input_pred.to_csv(os.path.join(util.modeldir,"consolidated_data","labelled_data.csv"))
			print("labelled data dumped")
		else:
			print("please specify the file to be predicted (-p)")


# labelled file does not exist


if __name__ == '__main__':
	main()


