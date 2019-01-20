#############################
##Training the models########
#############################
#############################

from sklearn.model_selection import train_test_split


# lets get the accuracy for using the feature matrix as one returned by word2vec

class training:

	def __init__(self,X,y):
		self.X_train , self.X_test , self.y_train, self.y_test = self.split_data() # split the dataset


	def split_data(self):
		# now tring with train_test split
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.25, random_state=0)
		return (X_train, X_test, y_train, y_test )

