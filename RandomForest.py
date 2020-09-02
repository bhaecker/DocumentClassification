import sys
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score

from .TransferLearning import fetch_data, loadmodel, retrain


def RandomForest_method(X,y,number_samples,model):

    if np.shape(X)[0] <= number_samples:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return(X,y,X_empty,y_empty)

    if type(model) == str:
        model = loadmodel(model)

    #train a Random Forest classifier on predictions and their correctness of the training set
    Xtrain,ytrain = fetch_data('train')
    #Xtrain,ytrain = Xtrain,ytrain

    ytrain_classes = np.argmax(ytrain, axis=1)

    ypred_train = model.predict(Xtrain)
    ypredtrain_classes = np.argmax(ypred_train,axis=1)

    #0 needs retraining, 1 is good to go
    decisions = np.array([0 if ytrain_classes[i] != ypredtrain_classes[i] else 1 for i in range(len(ytrain_classes))])

    #train RF on decisions
    clf = RandomForestClassifier(n_estimators = 500,max_depth=100, random_state=0)
    clf.fit(ypred_train, decisions)

    ypredunseen = model.predict(X)
    decisions_unseen = clf.predict(ypredunseen)
    Xneed, yneed = X[decisions_unseen == 0, :, :], y[decisions_unseen == 0]
    Xnoneed, ynoneed = X[decisions_unseen == 1, :, :], y[decisions_unseen == 1]

    # count samples which need retraining
    number_retrain = np.shape(Xneed)[0] #np.count_nonzero(decisions_unseen == 0)
    number_noretrain = np.shape(Xnoneed)[0]
    print('number_retrain:',number_retrain)
    print('number_noretrain:',number_noretrain)

    #chose as many retraining samples as we need
    #fill the rest with randomly choosen samples from the other group
    if number_retrain < number_samples:
        ind = np.arange(number_noretrain)
        np.random.shuffle(ind)
        Xwinner = np.concatenate((Xneed,Xnoneed[ind[:(number_samples-number_retrain)],:,:]), axis=0)
        Xloser = Xnoneed[ind[(number_samples-number_retrain):],:,:]
        ywinner = np.concatenate((yneed, ynoneed[ind[:(number_samples - number_retrain)]]), axis=0)
        yloser = ynoneed[ind[(number_samples - number_retrain):]]
    else:
        ind = np.arange(number_retrain)
        np.random.shuffle(ind)
        Xwinner = Xneed[ind[:number_samples], :, :]
        Xloser = np.concatenate((Xnoneed, Xneed[ind[number_samples:], :, :]), axis=0)
        ywinner = yneed[ind[:number_samples]]
        yloser = np.concatenate((ynoneed, yneed[ind[number_samples:]]), axis=0)

    #print(np.shape(Xwinner))
    #print(np.shape(ywinner))
    #print(np.shape(Xloser))
    #print(np.shape(yloser))

    return (Xwinner, ywinner, Xloser, yloser)


def RandomForestRegressor_pretraining(Xtrain,ytrain,basemodel,epochs_retrain_sample):
    if type(basemodel) == str:
        basemodel = loadmodel(basemodel)

    Xtest, ytest = fetch_data('test')
    Xtest, ytest = Xtest, ytest
    ypred = basemodel.predict(Xtest)

    ytest_flat = np.argmax(ytest, axis=1)
    ypred_flat = np.argmax(ypred, axis=1)
    base_accuracy = accuracy_score(ytest_flat, ypred_flat)

    score = np.empty(np.shape(Xtrain)[0])
    for idx in range(np.shape(Xtrain)[0]):

        image = np.expand_dims(Xtrain[idx,:,:], axis=0)
        #print(image)
        #print(ytrain[idx])
        model_new = retrain(basemodel,epochs_retrain_sample,1,Xtrain[idx:idx+1],ytrain[idx:idx+1])[0]
        ypred = model_new.predict(Xtest)
        ypred_flat = np.argmax(ypred, axis=1)
        new_accuracy = accuracy_score(ytest_flat, ypred_flat)
        score[idx] = new_accuracy-base_accuracy

    ypred_train = basemodel.predict(Xtrain)

    clf = RandomForestRegressor(n_estimators=100, max_depth=100, random_state=0)
    clf.fit(ypred_train, score)
    with open('RF', 'wb') as f:
        pickle.dump(clf, f)
    return(clf)



def RandomForest_fn(annotation_vector):
    Xtrain, ytrain = fetch_data('train')
    Xtrain, ytrain = Xtrain[:500], ytrain[:500]
    basemodel = 'model_100epochs'
    #RF = RandomForestRegressor_pretraining(Xtrain,ytrain,basemodel,1)
    with open('RF', 'rb') as f:
        RF = pickle.load(f)
    annotation_vector = annotation_vector.reshape(1,-1)
    score = RF.predict(annotation_vector)
    return(score[0])







