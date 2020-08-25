import sys
import math
import glob
import os
import numpy as np
import pandas as pd
import collections

import tensorflow as tf
#from tensorflow import keras as K
#from keras.models import model_from_json

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from .TransferLearning import fetch_data, fine_tune, retrain, savemodel, loadmodel
#from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn
from .ActiveLearning import seperation


DATA_DIRECTORY = 'Data'


def tester(Xtest,ytest,model):
    print('start evaluating')
    if type(model) == str:
        #K.clear_session()
        #model = tf.keras.models.load_model(model)
        model = loadmodel(model)

    loss = model.evaluate(Xtest, ytest, verbose=0)
    print('loss :'+ str(loss))

    ypred = model.predict(Xtest)

    ytest_flat = np.argmax(ytest, axis=1)
    ypred_flat = np.argmax(ypred, axis=1)
    accuracy = accuracy_score(ytest_flat, ypred_flat)
    print('acc :' + str(accuracy))
    print(confusion_matrix(ytest_flat, ypred_flat))

    return(accuracy,ypred)

def experiment(model_base,epochs_retrain,retrain_size,mini_batch_size,list_methods):

    '''
    :param model: initial fine tuned model, which is retrained
    :param epochs_retrain: number of epochs to retrain
    :param retrain_size: number of samples to retrain with
    :param mini_batch_size: batch size of the retraining
    :param list_methods: list of methods to choose the samples for retraining
    :return two dimensional array over methods and retrain cycles with performance measures
    '''

    if type(model_base) == str:
        model_base = loadmodel(model_base)

    Xtest, ytest = fetch_data('test')
    #Xtest, ytest = Xtest[:10], ytest[:10]
    Xtrain, ytrain = fetch_data('train')

    base_performance = tester(Xtest, ytest, model_base)[0]
    number_samples = 0
    df = pd.DataFrame([[number_samples]+[base_performance]*len(list_methods)], columns = ['number of samples'] + [str(method.__name__) for method in list_methods])

    Xunseen_orig, yunseen_orig = fetch_data('unseen')
    print(np.shape(Xunseen_orig))
    print(np.shape(Xtrain))
    #Xunseen_orig, yunseen_orig = Xunseen_orig[:10], yunseen_orig[:10]

    for method in list_methods:
        print('start',method.__name__)
        #Xtrain_new, ytrain_new = Xtrain, ytrain
        Xwinner, ywinner, Xloser, yloser = seperation(Xunseen_orig, yunseen_orig, model_base, retrain_size, method)

        # get the class distribution
        class_distribution = collections.Counter(np.where(ywinner == 1)[1])
        print(class_distribution)
        #new trainings batch consists of old training samples plus the new unseen ones
        Xtrain_new, ytrain_new = np.concatenate((Xtrain, Xwinner), axis=0), np.concatenate((ytrain, ywinner),axis=0)

        ##number_samples = retrain_size

        model_old = model_base
        index = 1
        #while number_samples <= np.shape(Xunseen_orig)[0]:
        while np.shape(Xwinner)[0] > 0:
            print(method.__name__,np.shape(Xwinner)[0])
            model_new = retrain(model_old,epochs_retrain,mini_batch_size,Xtrain_new, ytrain_new)[0]
            del model_old

            accuracy = tester(Xtest,ytest, model_new)[0]
            df.at[index, 'number of samples'] = np.shape(Xtrain_new)[0]-np.shape(Xtrain)[0]#number_samples
            df.at[index, str(method.__name__)] = accuracy

            Xwinner, ywinner, Xloser, yloser = seperation(Xloser, yloser, model_new, retrain_size, method)

            #get the class distribution
            #class_distribution = collections.Counter(np.where(ywinner == 1)[1])
            #print(np.shape(Xtrain_new))
            #print(np.shape(Xwinner))

            #print(np.shape(ytrain_new))
            #print(np.shape(ywinner))

            Xtrain_new, ytrain_new = np.concatenate((Xtrain_new, Xwinner), axis=0), np.concatenate((ytrain_new, ywinner), axis=0)
            index = index + 1
            #df.loc[len(df)] = [number_samples] + accuracy_list
            #number_samples = number_samples + retrain_size
            model_old = model_new
            print(df)

    df.to_csv('RESULTS.csv', index = False)
    print(df)
    return(df)