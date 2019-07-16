## functions for baseline models etc.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import preprocessing
import util


## @ return a trained svm model
def svm_wrapper(X_train, Y_train):
	svm = LinearSVC(C=1, verbose=1)
	svm.fit(X_train, Y_train)
	return svm


##  calculate the prediction for a model based on tfidf and svd
def cal_text_pred(test_data, model, tf_idf, svd):
	X_test = tf_idf.transform(test_data["tweetText"])
	X_test = svd.transform(X_test)  ## reduce the dimensionality
	y_pred = model.predict(X_test)
	return y_pred


def cal_user_pred(test_data, model):
	X_test, _ = preprocessing.prepare_user_features(test_data)
	y_pred = model.predict(X_test)
	return y_pred


## return trained models and their scores, for ensemble
def get_baseline_scores(X_train, X_test, Y_train, Y_test,cross_val=False):
	print("training the models")
	print("X_train shape",X_train.shape)
	print("X_test shape",X_test.shape)
	print("svm")
	svm = LinearSVC(C=1, verbose=1)
	svm.fit(X_train, Y_train)
	svm_pred = svm.predict(X_test)
	svm_f1 = precision_recall_fscore_support(Y_test,svm_pred, average=None)[2]  # mean f1
	
	print("random_forest")
	rf = RandomForestClassifier(n_estimators=100, max_depth=2,
	                            random_state=0)
	rf.fit(X_train, Y_train)
	rf_pred = rf.predict(X_test)
	rf_f1 = precision_recall_fscore_support(Y_test, rf_pred, average=None)[2]  # mean f1
	
	print("xgBoost")
	xgb = XGBClassifier()
	xgb.fit(X_train, Y_train)
	xgb_pred = xgb.predict(X_test)
	xgb_f1 = precision_recall_fscore_support(Y_test, xgb_pred, average=None)[2] # mean f1
	
	
	if cross_val == True:
		Y = np.array(list(Y_train) + list(Y_test))  ## for corss val
		X = np.concatenate((X_train, X_test), axis=0)  ## for cross_val
		svm_f1 = util.get_cross_val(svm, X, Y, n_splits=5)
		rf_f1 = util.get_cross_val(rf, X, Y, n_splits=5)
		xgb_f1 = util.get_cross_val(xgb, X, Y, n_splits=5)
		
		print('svm cross val score mean', svm_f1, '\n')
		print('rf cross val score mean', rf_f1, '\n')
		print('xgb corss val score mean', xgb_f1, '\n')
		
		models = {0: "svm", 1: "rf", 2: "xgb"}
		best_model_idx = np.argmax(
			[svm_f1.mean(), rf_f1.mean(), xgb_f1.mean()])  ## get the best performing model based on f1
	
		print("the best model", models[best_model_idx])

		
	
	print("baseline scores calculated")
	all_models = {
		'svm': [svm_pred, svm_f1, svm],
		'rf': [rf_pred, rf_f1, rf],
		'xgb': [xgb_pred, xgb_f1, xgb],
	}
	return (all_models)


def get_weighted_f1(Y_true, Y_pred):
	f_scores = precision_recall_fscore_support(Y_true, Y_pred)[2]
	supports = precision_recall_fscore_support(Y_true, Y_pred)[3]
	f1_num = 0
	for f_s, sup in zip(f_scores, supports):
		f1_num += (f_s * sup)
	f1 = f1_num / np.sum(supports)
	return (f1)
