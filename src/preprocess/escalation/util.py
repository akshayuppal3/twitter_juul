#  helper functions

import nltk
from numpy import zeros
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import re
nltk.download('wordnet')
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))
def get_tokens(sentence):
#     tokens = nltk.word_tokenize(sentence)  # now using tweet tokenizer
    tokens = tweet_tknzr.tokenize(sentence)
    tokens = [token.lower() for token in tokens]
    tokens = [token for token in tokens if (token not in stopwords and len(token) > 1)] ## remove punctuations
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
    return(len(a))

## cleaning files
def clean_text(text):
    text = re.sub(r'(https?://\S+)', "", text) ## remove url
    text = re.sub(r'(\@\w+)', "author",text)   ## remove @ mentions with author
    text = re.sub(r'(@)', "",text)             ## remove @ symbols
    text = re.sub(r'(author)',"",text)         ## remove author
    text = re.sub(r'(#)', "",text)             ## removing the hashtags signal
    text = re.sub(r'(RT )', "",text)         ## remove the retweet info as they dont convey any information
    text = re.sub(r'(^:)',"",text)
    text = text.rstrip()
    text = text.lstrip()
    return(text)


## returns the emnbedding matrix for the lstm model
def get_embedding_matrix(vocab_size,dimension,embedding_file,keras_tkzr):
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
            key = split_line[0] # the first word is the key
            value = np.array([float(val) for val in split_line[1:]])
            word2vec[key] = value
        return (word2vec)
    else:
        print("invalid fiel path")

