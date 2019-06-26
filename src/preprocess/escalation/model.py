
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

## @ return a trained svm model
def svm_wrapper(X_train, Y_train):
	svm = LinearSVC(C=1, verbose=1)
	svm.fit(X_train, Y_train)
	return svm


def get_baseline_scores(X_train ,X_test ,Y_train ,Y_test):
	
	print("training the models")
	print("svm")
	svm = LinearSVC(C=1 ,verbose=1)
	svm.fit(X_train ,Y_train)
	svm_pred = svm.predict(X_test)
	svm_score = precision_recall_fscore_support(Y_test ,svm_pred ,average=None)[2]  # return the f-score
	svm_f1 =  cross_val_score(svm, X_test, Y_test, cv=5 ,scoring='f1_macro').mean()
	print('svm cross val score mean' ,svm_f1 ,'\n')
	
	print("random_forest")
	rf = RandomForestClassifier(n_estimators=100, max_depth=2,
	                            random_state=0)
	rf.fit(X_train ,Y_train)
	rf_pred = rf.predict(X_test)
	rf_score = precision_recall_fscore_support(Y_test ,rf_pred ,average=None)[2]
	rf_f1  = cross_val_score(rf, X_test, Y_test, cv=5 ,scoring='f1_macro').mean()
	print('rf cross val score mean' ,rf_f1 ,'\n')
	
	print("xgBoost")
	xgb = XGBClassifier()
	xgb.fit(X_train, Y_train)
	xgb_pred = xgb.predict(X_test)
	xgb_score = precision_recall_fscore_support(Y_test ,xgb_pred ,average=None)[2]
	xgb_f1 =  cross_val_score(xgb, X_test, Y_test, cv=5 ,scoring='f1_macro').mean()
	print('xgb corss val score mean' ,xgb_f1 ,'\n')
	
	y_pred = [1 for x in range(len(Y_test))]
	print('  Classification Report:\n' ,classification_report(Y_test ,y_pred) ,'\n')
	maj_score = precision_recall_fscore_support(Y_test ,y_pred ,average=None)[2]
	
	models = {0 :[svm_pred ,svm ,"svm"], 1 :[rf_pred ,rf ,"rf"] ,2: [xgb_pred, xgb, "xgb"]}
	model_idx = np.argmax([svm_f1, rf_f1, xgb_f1])  ## get the best performing model
	
	print("selecting the best model", models[model_idx][2])
	
	print("job finished")
	all_scores = {
		'svm': [svm, svm_score],
		'rf': [rf, rf_score],
		'xg_boost': [xgb, xgb_score],
		'maj': [maj_score],
	}
	return (all_scores, models[model_idx])

## return cross_val mean score for each
def get_cross_val_score(train_data,Y_train,n_splits,nb_epoch):
    scores = []
    train_ids = list(train_data.index)
    kFold = StratifiedKFold(n_splits=n_splits)
    for train, test in kFold.split(train_ids,Y_train):
        X_train_user,_ = prepare_user_features(train_data.loc[train])
        X_test_user,_ = prepare_user_features(train_data.loc[test])

        encoded_docs = keras_tkzr.texts_to_sequences(train_data.loc[train]["tweetText"])
        X_train = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))
        encoded_docs = keras_tkzr.texts_to_sequences(train_data.loc[test]["tweetText"])
        X_test = (pad_sequences(encoded_docs, maxlen=max_len, padding='post'))

        history = model.fit([X_train,X_train_user],Y_train[train],validation_split=0.25 , nb_epoch = epoch,
                        verbose = 1,batch_size=32,class_weight= None,)
        training_plot(history)

        ## prediction
        temp = model.predict([X_test,X_test_user])
        y_pred = [np.argmax(value) for value in temp]  ## sigmoid
        f1 = precision_recall_fscore_support(Y_train[test],y_pred,average=None)[2]
        print(f1)
        print('  Classification Report:\n',classification_report(Y_train[test],y_pred),'\n')
        scores.append(f1)
    score1 = np.mean([ele[0] for ele in scores])
    score2 = np.mean([ele[1] for ele in scores])
    return (score1,score2)
