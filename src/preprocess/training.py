#############################
##Training the models########
#############################
#############################

from sklearn.model_selection import train_test_split
from models import *

# lets get the accuracy for using the feature matrix as one returned by word2vec

class training:
	# @ param X dependent variabel and y independent variable
	def __init__(self,X,y=None):
		self.X = X
		self.y = y

	def split_data(self):
		# now tring with train_test split
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=0)
		return (X_train, X_test, y_train, y_test )

	# (Perceptron, knn, dt) measures baseline acc and classification report @return void
	def train_baseline(self):
		self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()  # split the dataset
		print("Training the baseline models(Extra tree, logistic and Naive bayes)")
		print("\n" * 1)
		print("This might take some time :[estimated(42s)]")
		models = baseline_models(self.X_train, self.y_train)
		print("Test accuracy")
		best_model = prediction_models(models, self.X_test, self.y_test)
		name = best_model[1]
		model = best_model[0]
		model.fit(self.X,self.y)  # fiiting on our manually labelled 500 dataset as it is out input data
		return (model,name)

	def predict(self,model):
		print("predicting the labels for the input file")
		print("\n" * 1)
		print("This might take some time :[estimated(42s)]")
		ypred = model.predict(self.X)
		return ypred

