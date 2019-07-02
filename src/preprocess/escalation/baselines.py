## functions for baseline models etc.

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

import preprocessing


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
def get_baseline_scores(X_train, X_test, Y_train, Y_test):
	print("training the models")
	print("svm")
	svm = LinearSVC(C=1, verbose=1)
	svm.fit(X_train, Y_train)
	svm_pred = svm.predict(X_test)
	# svm_score = precision_recall_fscore_support(Y_test, svm_pred, average=None)[2]  # return the f-score
	svm_f1 = cross_val_score(svm, X_test, Y_test, cv=5, scoring='f1_macro').mean()
	print('svm cross val score mean', svm_f1, '\n')
	
	print("random_forest")
	rf = RandomForestClassifier(n_estimators=100, max_depth=2,
	                            random_state=0)
	rf.fit(X_train, Y_train)
	rf_pred = rf.predict(X_test)
	# rf_score = precision_recall_fscore_support(Y_test, rf_pred, average=None)[2]
	rf_f1 = cross_val_score(rf, X_test, Y_test, cv=5, scoring='f1_macro').mean()
	print('rf cross val score mean', rf_f1, '\n')
	
	print("xgBoost")
	xgb = XGBClassifier()
	xgb.fit(X_train, Y_train)
	xgb_pred = xgb.predict(X_test)
	# xgb_score = precision_recall_fscore_support(Y_test, xgb_pred, average=None)[2]
	xgb_f1 = cross_val_score(xgb, X_test, Y_test, cv=5, scoring='f1_macro').mean()
	print('xgb corss val score mean', xgb_f1, '\n')
	
	y_pred = [1 for x in range(len(Y_test))]
	# print('  Classification Report:\n', classification_report(Y_test, y_pred), '\n')
	maj_score = precision_recall_fscore_support(Y_test, y_pred, average=None)[2]
	
	models = {0: "svm", 1: "rf", 2: "xgb"}
	best_model_idx = np.argmax([svm_f1, rf_f1, xgb_f1])  ## get the best performing model based on f1
	
	print("the best model", models[best_model_idx])
	
	print("job finished")
	all_models = {
		'svm': [svm_pred, svm_f1, svm],
		'rf': [rf_pred, rf_f1, rf],
		'xgb': [xgb_pred, xgb_f1, xgb],
		'maj': [maj_score],
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
