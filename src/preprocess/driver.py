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

# Suppress warning
def warn(*args, **kwargs):
	pass


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
			df_input = pd.read_csv(filename)  # input data
			print("generating sentecnes might take some time (5-10min)")
			sentences_input = util.get_sentences(df_input, 'tweetText')
			model = Word2Vec(sentences_input, size=100, min_count=2)
			w2v = dict(zip(model.wv.index2word, model.wv.syn0))
			# dump the w2v embeddings
			w2v_file = os.path.join(util.modeldir,"w2v.pkl")
			with open(w2v_file,"wb") as f:
				pickle.dump(w2v,f)
		else:
			print("specify the input file")

	# training the model
	elif args['function'] == 'train':
		# user the labelled file to get classification accuracy with different models
		w2v_filename = os.path.join(util.modeldir + "w2v.pkl")
		if (os.path.exists(w2v_filename)):  ## check if embedding exists
			w2v = pickle.load(open(w2v_filename, "rb"))
			if (w2v):
				if (os.path.exists(os.path.join(util.modeldir + "df_timeline_lbl.pkl"))):
					lbl_file_path = os.path.join(util.modeldir + "df_timeline_lbl.pkl")
					print("loading the labelled file")
					df_lbl = pickle.load(open(lbl_file_path),"rb")
					print("\n************")
					print(len(df_lbl))   ## delete
					pre = Preprocess(w2v,df_lbl)
					sentences = util.get_sentences(df_lbl, 'tweetText')
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
					print("taking the" + str(name) + " to predict on the input")
				else:
					print("unzip the labelled file or the labelled file does not exist")
			else:
				print("w2v embeddings file does not exist")

	# predicting the model
	elif args['function'] == 'predict':
		if (args['inputFile']):
			input_file_path = args['inputFile']
			df_input = pd.read_csv(input_file_path)
			w2v_filename = os.path.join(util.modeldir + "w2v.pkl")
			if (os.path.exists(w2v_filename)):  ## check if embedding exists
				w2v = pickle.load(open(w2v_filename, "rb"))
				if (w2v):
					print("\n************")
					print(len(df_input))  ## delete
					pre = Preprocess(w2v, df_input)
					sentences = util.get_sentences(df_input, 'tweetText')
					## getting the features of the labelled data
					print("getting the features of labelled data")
					X = pre.get_X(sentences)
					X = preprocessing.normalize(X, norm='l1')
					# get the model from the previous training process
					train_model_path = os.path.join(util.modeldir,"train_model.pkl")
					train_model = pickle.load(open(train_model_path),"rb")
					train = training(X)
					y_pred = train.predict(train_model)
					## get the no of labelled as 1, 2 and 3
					print("no of labelled as 3",len(y_pred[y_pred == 3]))
					print("no of labelled as 2", len(y_pred[y_pred] ==2))
					print("no of labelled as 1", len(y_pred[y_pred] ==1))
					df_input['label'] = y_pred
					# dump the final full labelled dataset
					input_labelled_path = os.path.join(util.modeldir,"df_timeline_lbl.pkl")
					with open(input_labelled_path,"rb") as f:
						pickle.dump(df_input,f)
			else:
				print("w2v embeddings file does not exist")
		else:
			print("please specify the input file")

	## extract the poly and mono users
	elif args['funtion'] == 'users':
		# get the labelled file
		if (os.path.exists(os.path.join(util.modeldir + "df_timeline_lbl.pkl"))):
			lbl_file_path = os.path.join(util.modeldir + "df_timeline_lbl.pkl")
			df_input = pd.read_csv(lbl_file_path)
			# open the weed list
			weed_words = pickle.load(open(os.path.join(util.modeldir,"weed_words.pkl"),"rb"))
			weed_words = [(" " + word + " ") for word in weed_words]
			pattern = "|".join(weed_words)
			print("extracting the pattern")
			df_tweet_weeds = df_input[df_input['tweetText'].str.contains(pattern, case=False)]
			index = df_tweet_weeds.index   # the file after filtering have the index contained within
			df_weeds = df_input.loc[index]
			poly_users = list(set(list(df_weeds.loc[df_weeds.label_pred == 3]['userID'])))
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


# labelled file does not exist


if __name__ == '__main__':
	main()


