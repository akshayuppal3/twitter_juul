# check for w2v in models else create and dump

import pickle
import warnings
import os
import argparse
import pandas as pd
from gensim.models import Word2Vec
from preprocess import Preprocess
from sklearn import preprocessing
from training import training
import util
from tqdm import tqdm
# Suppress warning
def warn(*args, **kwargs):
	pass

# df_train_lbl is the 500 lablled data
# df_timeline_lbl is the labelleing of teh entire input file


def main():
	warnings.warn = warn
	parser = argparse.ArgumentParser(description="getting the poly and the mono users")
	parser.add_argument("-i",'--inputFile',help="specify the Dataframe file to get(poly/mono) users",required=False)
	parser.add_argument("-f",'--function',help="specify the option as ( w2v / train / predict/ users)", default='train')  # option for train or prediction
	args = vars(parser.parse_args())

	# creating embeddings from the input file
	if args['function'] == 'w2v':
		if (args['inputFile']):
			filename = args['inputFile']
			if filename.endswith('.pkl'):
				df_input = pickle.load(open(filename,"rb"))
			elif filename.endswith('.csv'):
				df_input = pd.read_csv(filename)  # input data
			print("generating sentecnes might take some time (5-10min)")
			sentences_input = util.get_sentences(df_input, 'tweetText')
			sentences_path = os.path.join(util.modeldir, "sentences.pkl")
			with open(sentences_path,"wb") as f:
				pickle.dump(sentences_input,f)
			print("pickle the sentences")
			model = Word2Vec(sentences_input, size=100, min_count=2)
			w2v = dict(zip(model.wv.index2word, model.wv.syn0))
			# dump the w2v embeddings
			w2v_file = os.path.join(util.modeldir,"w2v.pkl")
			with open(w2v_file,"wb") as f:
				pickle.dump(w2v,f)
			print("dumping the w2v model finished")
		else:
			print("specify the input file")

	# training the model
	elif args['function'] == 'train':
		# user the labelled file to get classification accuracy with different models
		w2v_filename = os.path.join(util.modeldir + "w2v.pkl")
		if (os.path.exists(w2v_filename)):  ## check if embedding exists
			w2v = pickle.load(open(w2v_filename, "rb"))
			if (w2v):
				if (os.path.exists(os.path.join(util.modeldir + "df_train_lb.pkl"))):
					lbl_file_path = os.path.join(util.modeldir + "df_train_lb.pkl")
					print("loading the labelled file")
					df_lbl = pickle.load(open(lbl_file_path,"rb"))
					print("\n************")
					print(len(df_lbl))   ## delete
					print(type(df_lbl))
					sentences = util.get_sentences(df_lbl, 'tweetText')
					pre = Preprocess(w2v,df_lbl)
					## getting the features of the labelled data
					print("getting the features of labelled data")
					X = pre.get_X(sentences)
					y = list(df_lbl.label)
					# normalize the features
					X = preprocessing.normalize(X, norm='l1')
					y = [int(i) for i in y]
					train = training(X,y)
					best_model = train.train_baseline()   # it returns a tuple (model,name)
					model = best_model[0]
					name = best_model[1]
					train_model_path = os.path.join(util.modeldir,'train_model.pkl')
					with open(train_model_path,"wb") as f:
						pickle.dump(model,f)
					print("taking the " + str(name) + " classifier to predict on the input")
				else:
					print("unzip the labelled file or the labelled file does not exist")
			else:
				print("w2v embeddings file does not exist")

	# predicting the model
	elif args['function'] == 'predict':
		if (args['inputFile']):
			input_file_path = args['inputFile']
			if input_file_path.endswith('.pkl'):
				df_input = pickle.load(open(input_file_path,"rb"))
			elif input_file_path.endswith('.csv'):
				df_input = pd.read_csv(input_file_path)
			w2v_filename = os.path.join(util.modeldir + "w2v.pkl")
			if (os.path.exists(w2v_filename)):  ## check if embedding exists
				w2v = pickle.load(open(w2v_filename, "rb"))
				if (w2v):
					print("\n************")
					print(len(df_input))  ## delete
					pre = Preprocess(w2v, df_input)
					sentences_path = os.path.join(util.modeldir, "sentences.pkl")
					if (os.path.exists(sentences_path)):
						sentences = pickle.load(open(sentences_path),"rb")
					else:
						sentences = util.get_sentences(df_input, 'tweetText')
						with open(sentences_path,"wb") as f:
							pickle.dump(sentences, f)
					## getting the features of the labelled data
					print("getting the features of labelled data")
					X = pre.get_X(sentences)
					# X = preprocessing.normalize(X, norm='l1')
					# get the model from the previous training process
					train_model_path = os.path.join(util.modeldir,"train_model.pkl")
					train_model = pickle.load(open(train_model_path,"rb"))
					train = training(X)
					y_pred = train.predict(train_model)
					## get the no of labelled as 1, 2 and 3
					print("no of labelled as 3",len(y_pred[y_pred == 3]))
					print("no of labelled as 2", len(y_pred[y_pred == 2]))
					print("no of labelled as 1", len(y_pred[y_pred == 1]))
					df_input['label'] = y_pred
					# dump the final full labelled dataset
					input_labelled_path = os.path.join(util.modeldir,"df_timeline_lbl.pkl")
					with open(input_labelled_path,"wb") as f:
						pickle.dump(df_input,f)
					print("labelled data dumped")
			else:
				print("w2v embeddings file does not exist")
		else:
			print("please specify the input file")

	## extract the poly and mono users
	elif args['function'] == 'users':
		# get the labelled file
		if (os.path.exists(os.path.join(util.modeldir + "df_timeline_lbl.pkl"))):
			lbl_file_path = os.path.join(util.modeldir + "df_timeline_lbl.pkl")
			df_input = pickle.load(open(lbl_file_path, "rb"))
			# open the weed list
			weed_words = pickle.load(open(os.path.join(util.modeldir,"weed_words.pkl"),"rb"))
			weed_words = [(" " + word + " ") for word in weed_words]
			pattern = "|".join(weed_words)
			print("extracting the pattern")
			df_tweet_weeds = df_input[df_input['tweetText'].str.contains(pattern, case=False)]
			index = df_tweet_weeds.index   # the file after filtering have the index contained within
			df_weeds = df_input.loc[index]
			poly_users = list(set(list(df_weeds.loc[df_weeds.label == 3]['userID'])))
			poly_path = os.path.join(util.modeldir, "poly_users.pkl")
			with open(poly_path,"wb") as f:
				pickle.dump(poly_users,f)
			total_users = list(df_input.userID.unique())
			poly_length = len(poly_users)
			total_users_length = len(total_users)
			mono_length = total_users_length - poly_length
			print("total users = ", total_users_length)
			print("no of poly users = ",poly_length)
			print("no of mono users = ", mono_length)
			print("% of poly users is ", poly_length / total_users_length )
			print("% of mono users is ", mono_length/ total_users_length )

			# classify the poly further as poly-1(have used drug before), 2 (have used after) ,3 (have used at both times)

		else:
			print("unzip the labelled file or the file does not exist")

	elif args['function'] == 'poly_s':
		# get the labelled file
		if (os.path.exists(os.path.join(util.modeldir + "df_timeline_lbl.pkl"))):
			lbl_file_path = os.path.join(util.modeldir + "df_timeline_lbl.pkl")
			df_input = pickle.load(open(lbl_file_path, "rb"))
			poly_path = os.path.join(util.modeldir + "poly_users.pkl")
			if(os.path.exists(poly_path)):
				poly_users = pickle.load(open(poly_path),"rb")
				# open the weed list
				weed_words = pickle.load(open(os.path.join(util.modeldir, "weed_words.pkl"), "rb"))
				weed_words = [(" " + word + " ") for word in weed_words]
				pattern_weed = "|".join(weed_words)
				pattern_juul = 'juul'
				poly_user1 = list()
				poly_user2 = list()
				poly_user3 = list()
				poly_und = list()
				total_users = list(df_input.userID.unique())
				for user in tqdm(poly_users):
					user_tweets = df_input.loc[df_input.userID == user]
					user_tweets.sort_values(by='tweetCreatedAt', ascending=True, inplace=True)  # sort by tweet created at
					juul_tweets = user_tweets[user_tweets['tweetText'].str.contains(pattern_juul, case=False)]
					juul_tweets.reset_index(drop=True, inplace=True)
					if (len(juul_tweets) > 0):
						time_j = pd.to_datetime(juul_tweets.head(1)['tweetCreatedAt'].values[0])  # getting the tweet with
					else:
						time_j = None
					weed_tweets = user_tweets[user_tweets['tweetText'].str.contains(pattern_weed, case=False)]
					weed_tweets_user = weed_tweets[
						weed_tweets.label_pred == 3]  # .first_valid_index()  # getting teh first creation of juul tweet
					times_w = None
					if (len(weed_tweets_user) > 0):
						times_w = pd.to_datetime(
							list(weed_tweets['tweetCreatedAt']))  # . weed_tweets.loc[index_w]['tweetCreatedAt'])
					if (time_j != None and len(times_w) > 0):
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
				print("% of pol1 users = ", len(poly_user1)/len(total_users))
				print("\n")
				print("% of pol2 users = ", len(poly_user2) / len(total_users))
				print("\n")
				print("% of pol3 users = ", len(poly_user3) / len(total_users))
				print("\n")
				print("% of undefined users = ", len(poly_und) / len(total_users))
			else:
				print("poly users not found- run the users option")

# labelled file does not exist


if __name__ == '__main__':
	main()


