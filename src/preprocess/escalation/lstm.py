## func related to lstm


from keras.callbacks import Callback

def training_plot(history):
	# Plot training & validation accuracy values
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Validation'], loc='upper left')
	plt.show()
	
	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()


def cal_lstm_pred(test_data, Y_test, model, keras_tkzr, max_len):
	## encoding the test data
	encoded_docs = keras_tkzr.texts_to_sequences(test_data["tweetText"])
	X_test = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
	X_test_user, _ = prepare_user_features(test_data)
	## calculate the model predictions
	temp = model.predict([X_test, X_test_user])
	y_pred = [np.argmax(value) for value in temp]  ## sigmoid
	return y_pred


## handle two different inputs and then concatenate them (user and text features)
## input = [words_in,user_in]
def create_model(max_len, user_feature_len, vocalb_size, dimension, embedding_matrix):
	## handle text features..
	words_in = Input(shape=(max_len,))
	emb_word = Embedding(vocalb_size, dimension, weights=[embedding_matrix], input_length=max_len)(words_in)
	lstm_word = Bidirectional(LSTM(100, return_sequences=False, dropout=0.50, kernel_regularizer=regularizers.l2(0.01)),
	                          merge_mode='concat')(emb_word)
	lstm_word = Dense(user_feature_len, activation='relu')(lstm_word)
	
	## takes the user features as input
	user_input = Input(shape=(user_feature_len,))
	
	## concatenate both of the features
	modelR = concatenate([lstm_word, user_input])
	# modelR = SpatialDropout1D(0.1)(modelR)
	output = Dense(2, activation='softmax')(modelR)
	model = Model([words_in, user_input], output)
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model



class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.array([np.argmax(value) for value in val_predict])
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print ("— val_f1: %f — val_precision: %f — val_recall %f"%(_val_f1, _val_precision, _val_recall))
        print('  Classification Report:\n',classification_report(val_targ,val_predict),'\n')
        return
