# check for w2v in models else create and dump


from setup import setup_env
import pickle
import nltk
import warnings
import os
import util
import argparse
import pandas as pd
from gensim.models import Word2Vec
from w2v import W2v
from preprocess import Preprocess
from sklearn import preprocessing
from training import training

# Suppress warning
def warn(*args, **kwargs):
	pass



def main():
	if (os.path.exists(os.path.join(util.modeldir + "df_timeline_lbl.pkl"))):
		lbl_file_path = os.path.join(util.modeldir + "sentences_timeline.pkl")
		parser = argparse.ArgumentParser(description="getting the poly and the mono users")
		parser.add_argument("-i",'--inputFile',help="specify the Dataframe file to get(poly/mono) users",required=True)
		parser.add_argument("-p","--annotated",help="path of the manually labelled file for tweet classification",default=lbl_file_path)
		parser.add_argument("-o",'--boolw2v',help="create embeddings from data(yes, no)",default=False,type=util.str2bool)
		args = vars(parser.parse_args())
		warnings.warn = warn
		setup_env()  # download necessary nltk packages
		stopwords = nltk.corpus.stopwords.words('english')
		w2v = False
		if (args['inputFile']):
			filename = args['inputFile']
			df_timeline = pd.read_csv(filename)  # input data
			print("generating sentecnes might take some time (5-10min)")
			sentences_timeline = util.get_sentences(df_timeline, 'tweetText')
			if not(args['boolw2v']):
				w2v_filename = os.path.join(util.modeldir + "w2v.pkl")
				if (os.path.exists(w2v_filename)):
					w2v = pickle.load(open(w2v_filename,"rb"))
				else:
					print(w2v_filename," does not exist - specify the option as True (it will create the embeddings)")
			else:
				model = Word2Vec(sentences_timeline, size=100, min_count=2)
				w2v = dict(zip(model.wv.index2word, model.wv.syn0))

			# user the labelled file to get classification accuracy with different models
			if (w2v):
				df_lbl = pd.read_csv(lbl_file_path)
				preproces = Preprocess(w2v,df_lbl)
				sentences = util.get_sentences(df_lbl, 'tweetText')
				X = preproces.get_X(sentences)
				y = list(df_lbl.label)
				# normalize the features
				X = preprocessing.normalize(X, norm='l1')
				y = [int(i) for i in y]
				train = training(X,y)
				train.train_baseline()
				print("taking the best model to predict on the input (df_timeline)")





