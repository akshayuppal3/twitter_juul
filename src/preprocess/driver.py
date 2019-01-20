
# check for w2v in models else create and dump


from setup import setup_env
import pickle
import nltk
import warnings

#Suppress warning
def warn(*args, **kwargs):
    pass

warnings.warn = warn
setup_env()  # download necessary nltk packages
stopwords = nltk.corpus.stopwords.words('english')
w2v = pickle.load(open("/Users/akshayuppal/Desktop/thesis/twitter_juul/models/w2v.pkl","rb"))