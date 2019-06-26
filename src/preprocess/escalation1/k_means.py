import operator

import gensim
import numpy as np
from sklearn import cluster
from tqdm import tqdm


### k means funtion

## creating X(input array for Kmeans) based on word2vec dimesnions
def get_word2vec_array(model):
	w2v = dict()
	vectors = []
	for ele in (model.wv.vocab):
		w2v[ele] = (model.wv[ele])
		vectors.append(model.wv[ele])
	X = np.array(vectors)
	return X


## get words in each cluster
def get_words_cluster(assigned_cluster, model):
	cluster_dict = {}
	for i, word in enumerate(list(model.wv.vocab)):
		index = assigned_cluster[i]
		if (index in cluster_dict):
			cluster_dict[index].append(str(word))
		else:
			cluster_dict[index] = [str(word)]
	return cluster_dict


def get_cluster_vector(tokens, cluster_dict):
	vector_dict = {}
	## iterate over all of the clusters
	if tokens:
		for index, cluster_words in cluster_dict.items():
			temp = float(len(set(tokens).intersection(set(cluster_words))) / len(tokens))
			vector_dict[index] = temp
		temp = [value for key, value in sorted(vector_dict.items(), key=operator.itemgetter(0))]  ## order by cluster
		vector_final = np.array(temp)
	else:  ## in case its blank (blank tweet by user)
		vector_final = np.zeros(len(cluster_dict))
	return vector_final


## get unique elements for each element
def get_unigrams(text):
	return (list(set(text)))

## main driver funtion
def get_w2v_kmeans_vector(sentences, n_clusters):
	print("running word2vec")
	model = gensim.models.Word2Vec(
		sentences,
		size=100,
		window=10,
		min_count=1,
		workers=10,
		iter=10)
	
	## get word2vec array for based on vocabulary
	print("getting word2vec array")
	X = get_word2vec_array(model)
	
	print("running K-means algorithm")
	## performing Kmeans
	kmeans = cluster.KMeans(n_clusters=n_clusters, verbose=3)
	kmeans.fit(X)
	labels = kmeans.labels_
	
	print("getting cluster dictionary for word")
	## getting the cluster dictionary (words in each cluster for vocabulary words)
	cluster_dict = get_words_cluster(labels, model)
	
	print("getting unigrams for sentences")
	## get unique unigrams from the user
	unigrams = []
	for elements in sentences:
		unigrams.append(get_unigrams(elements))
	
	print("getting the final array")
	## creating the final X array
	X_final = np.empty((0, n_clusters))
	for tokens in tqdm(unigrams):
		cluster_vector = get_cluster_vector(tokens, cluster_dict)
		#     print(np.sum(cluster_vector))  ## sanity check
		X_final = np.append(X_final, cluster_vector.reshape(1, -1), axis=0)
	return X_final
