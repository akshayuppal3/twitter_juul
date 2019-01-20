# check for w2v in models else create and dump


from setup import setup_env
import pickle
import nltk
import warnings
from models import *


# Suppress warning
def warn(*args, **kwargs):
	pass


warnings.warn = warn
setup_env()  # download necessary nltk packages
stopwords = nltk.corpus.stopwords.words('english')
w2v = pickle.load(open("/Users/akshayuppal/Desktop/thesis/twitter_juul/models/w2v.pkl", "rb"))


def train_baseline(X_train, Y_train, X_dev, Y_dev, X_test, Y_test):
	print("Training the baseline models(decision tree, knn, perceptron)")
	print("\n" * 1)
	print("This might take some time :[estimated(42s)]")
	models = baseline_models(X_train, Y_train)
	print("Development accuracy")
	prediction_models(models, X_dev, Y_dev)
