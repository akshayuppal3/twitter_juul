import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
# grid search
from sklearn.model_selection import GridSearchCV


def baseline_models(X_train, y_train):
	models = {}
	etree = extra_tree(X_train, y_train)
	logistic = logistic_regression(X_train, y_train)
	nb = naive_bayes(X_train, y_train)
	models = {"extra_tree": etree, "logistic": logistic, "naive_bayes": nb}
	return (models)


def extra_tree(X_train, y_train):  # using etree as gave best scre
	etree = ExtraTreesClassifier(n_estimators=200)  # @TODO grid search
	etree.fit(X_train, y_train)


def naive_bayes(X_train, y_train):
	NB = GaussianNB()
	NB.fit(X_train, y_train)


def logistic_regression(X_train, y_train):
	Lr = LogisticRegression(random_state=0, solver='lbfgs',
							multi_class='multinomial')
	Lr.fit(X_train, y_train)


def prediction(model, X, y):
	Y_pred = model.predict(X)
	score = accuracy_score(y, Y_pred)
	return (score)


def classification(Y_dev, Y_pred):
	confusion_matrix(Y_dev, Y_pred)
	print('  Classification Report:\n', classification_report(Y_dev, Y_pred), '\n')


def prediction_models(models, X, Y_test):
	for name, model in models.items():
		Y_pred = model.predict(X)
		score = accuracy_score(Y_test, Y_pred)
		print("accuarcy ", name, ":", score)
		classification(Y_test, Y_pred)
