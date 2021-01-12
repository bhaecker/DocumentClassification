import sys
import math
import glob
import os
import numpy as np
import pandas as pd
#import collections
#import copy

import tensorflow as tf
from tensorflow.keras.models import load_model
#from tensorflow import keras as K
#from keras.models import model_from_json

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from .TransferLearning import fetch_data, fine_tune, retrain, concate,retrain_early_stopping#, savemodel, loadmodel
#from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn
from .ActiveLearning import seperation
#from .RandomForest import RandomForestRegressor_pretraining


def tester(Xtest,ytest,model):
    print('start evaluating')
    if type(model) == str:
        model = load_model(model)

    loss = model.evaluate(Xtest, ytest, verbose=0)
    print('loss :'+ str(loss[0]))

    ypred = model.predict(Xtest)

    ytest_flat = np.argmax(ytest, axis=1)
    ypred_flat = np.argmax(ypred, axis=1)
    accuracy = accuracy_score(ytest_flat, ypred_flat)
    print('acc :' + str(accuracy))
    print(confusion_matrix(ytest_flat, ypred_flat))

    return(accuracy,ypred)



def experiment_accumulated(model_base_str,epochs_retrain,retrain_size,mini_batch_size,list_methods):

    '''
    :param model: initial fine tuned model, which is retrained, provided as a string
    :param epochs_retrain: number of epochs to retrain
    :param retrain_size: number of samples to retrain with
    :param mini_batch_size: batch size of the retraining
    :param list_methods: list of methods to choose the samples for retraining
    :return two dimensional array over methods and retrain cycles with performance measures
    '''

    #if type(model_base) == str:
        #model_base = loadmodel(model_base)
     #   model_base = load_model(model_base)
    if type(model_base_str) != str:
        exit('please provide name of model as string')


    Xtest, ytest = fetch_data('test')
    Xtrain, ytrain = fetch_data('train')

    base_performance = tester(Xtest, ytest, model_base_str)[0]
    number_samples = 0
    df = pd.DataFrame([[number_samples]+[base_performance]*len(list_methods)], columns = ['number of samples'] + [str(method.__name__) for method in list_methods])

    Xunseen_orig, yunseen_orig = fetch_data('unseen')

    #shuffle the unseen data
    rng_state = np.random.get_state()
    np.random.shuffle(Xunseen_orig)
    np.random.set_state(rng_state)
    np.random.shuffle(yunseen_orig)

    #Xunseen_orig, yunseen_orig = Xunseen_orig[:200], yunseen_orig[:200]

    for method in list_methods:
        print('start',method.__name__)
        Xwinner, ywinner, Xloser, yloser = seperation(Xunseen_orig, yunseen_orig, model_base_str, retrain_size, method)

        #new trainings batch consists of old training samples plus the new unseen ones
        Xtrain_new = np.concatenate((Xtrain, Xwinner), axis=0)
        ytrain_new = np.concatenate((ytrain, ywinner),axis=0)
        ##number_samples = retrain_size

        #model_old = model_base
        index = 1
        #while number_samples <= np.shape(Xunseen_orig)[0]:
        while np.shape(Xwinner)[0] > 0:
            print(method.__name__,'at '+str(100*(np.shape(Xtrain_new)[0] - np.shape(Xtrain)[0])/np.shape(Xunseen_orig)[0])+' %')
            #just for debugging
            #if index == 1:
             #   model_new = retrain(model_base,epochs_retrain,mini_batch_size,Xtrain_new, ytrain_new)[0]
              #  print(tester(Xtest,ytest, model_new)[0])
            model_base = load_model(model_base_str)
            model_new = retrain(model_base,epochs_retrain,mini_batch_size,Xtrain_new, ytrain_new)[0]

            #del model_old

            accuracy = tester(Xtest,ytest, model_new)[0]
            df.at[index, 'number of samples'] = np.shape(Xtrain_new)[0]-np.shape(Xtrain)[0]#number_samples
            df.at[index, str(method.__name__)] = accuracy

            Xwinner, ywinner, Xloser, yloser = seperation(Xloser, yloser, model_new, retrain_size, method)

            Xtrain_new, ytrain_new = np.concatenate((Xtrain_new, Xwinner), axis=0), np.concatenate((ytrain_new, ywinner), axis=0)
            index = index + 1

            del model_new, model_base
            #model_old = model_new

            print(df)

    df.to_csv('RESULTS.csv', index = False)
    print(df)
    return(df)

def experiment_single(model_base_str,epochs_retrain,retrain_size,mini_batch_size,list_methods):

    '''
    :param model: initial fine tuned model, which is retrained, provided as a string
    :param epochs_retrain: number of epochs to retrain
    :param retrain_size: number of samples to retrain with
    :param mini_batch_size: batch size of the retraining
    :param list_methods: list of methods to choose the samples for retraining
    :return two dimensional array over methods and retrain cycles with performance measures
    '''

    #if type(model_base) == str:
        #model_base = loadmodel(model_base)
     #   model_base = load_model(model_base)
    if type(model_base_str) != str:
        exit('please provide name of model as string')

    Xtest, ytest = fetch_data('test')
    Xtrain, ytrain = fetch_data('train')
    print('right data fetched')

    base_performance = round(tester(Xtest, ytest, model_base_str)[0],2)
    df = pd.DataFrame([[0]+[base_performance]*len(list_methods)], columns = ['fraction used'] + [str(method.__name__) for method in list_methods])

    Xunseen_orig, yunseen_orig = fetch_data('unseen')

    #shuffle the unseen data
    rng_state = np.random.get_state()
    np.random.shuffle(Xunseen_orig)
    np.random.set_state(rng_state)
    np.random.shuffle(yunseen_orig)

    #Xunseen_orig, yunseen_orig = Xunseen_orig[:200], yunseen_orig[:200]

    for method in list_methods:
        print('start',method.__name__)
        Xwinner, ywinner, Xloser, yloser = seperation(Xunseen_orig, yunseen_orig, model_base_str, retrain_size, method)

        #new trainings batch consists of old training samples plus the new unseen ones
        #Xtrain_new = np.concatenate((Xtrain, Xwinner), axis=0)
        #ytrain_new = np.concatenate((ytrain, ywinner),axis=0)

        Xtrain_new = concate(Xtrain, Xwinner)
        ytrain_new = concate(ytrain, ywinner)


        index = 1
        print('training data concatenated')
        #while number_samples <= np.shape(Xunseen_orig)[0]:
        while np.shape(Xwinner)[0] > 0:
            #print(method.__name__,'at '+str(100*(np.shape(Xtrain_new)[0] - np.shape(Xtrain)[0])/np.shape(Xunseen_orig)[0])+' %')
            #just for debugging
            #if index == 1:
             #   model_new = retrain(model_base,epochs_retrain,mini_batch_size,Xtrain_new, ytrain_new)[0]
              #  print(tester(Xtest,ytest, model_new)[0])
            model_base = load_model(model_base_str)
            print('model loaded')
            model_new = retrain(model_base,epochs_retrain,mini_batch_size,Xtrain_new, ytrain_new)[0]

            accuracy = tester(Xtest,ytest, model_new)[0]
            df.at[index, 'fraction used'] = round(1-np.shape(Xloser)[0]/np.shape(Xunseen_orig)[0],2)#number_samples
            df.at[index, str(method.__name__)] = accuracy
            print(df)
            Xwinner, ywinner, Xloser, yloser = seperation(Xloser, yloser, model_new, retrain_size, method)

            #Xtrain_new, ytrain_new = np.concatenate((Xtrain, Xwinner), axis=0), np.concatenate((ytrain, ywinner), axis=0)
            Xtrain_new, ytrain_new = concate(Xtrain, Xwinner), concate(ytrain, ywinner)

            index = index + 1

            del model_new, model_base



    df.to_csv('RESULTS.csv', index = False)
    print(df)
    return(df)

def experiment_single_earlystopping(model_base_str,epochs_retrain,retrain_size,mini_batch_size,list_methods):

    '''
    :param model: initial fine tuned model, which is retrained, provided as a string
    :param epochs_retrain: number of epochs to retrain
    :param retrain_size: number of samples to retrain with
    :param mini_batch_size: batch size of the retraining
    :param list_methods: list of methods to choose the samples for retraining
    :return two dimensional array over methods and retrain cycles with performance measures
    '''

    #if type(model_base) == str:
        #model_base = loadmodel(model_base)
     #   model_base = load_model(model_base)
    if type(model_base_str) != str:
        exit('please provide name of model as string')

    Xtest, ytest = fetch_data('test')
    Xtrain, ytrain = fetch_data('train')
    print('right data fetched')

    base_performance = round(tester(Xtest, ytest, model_base_str)[0],2)
    df = pd.DataFrame([[0]+[base_performance]*len(list_methods)], columns = ['fraction used'] + [str(method.__name__) for method in list_methods])

    Xunseen_orig, yunseen_orig = fetch_data('unseen')

    #shuffle the unseen data
    rng_state = np.random.get_state()
    np.random.shuffle(Xunseen_orig)
    np.random.set_state(rng_state)
    np.random.shuffle(yunseen_orig)

    #Xunseen_orig, yunseen_orig = Xunseen_orig[:200], yunseen_orig[:200]

    for method in list_methods:
        print('start',method.__name__)
        Xwinner, ywinner, Xloser, yloser = seperation(Xunseen_orig, yunseen_orig, model_base_str, retrain_size, method)

        #new trainings batch consists of old training samples plus the new unseen ones
        #Xtrain_new = np.concatenate((Xtrain, Xwinner), axis=0)
        #ytrain_new = np.concatenate((ytrain, ywinner),axis=0)

        Xtrain_new = concate(Xtrain, Xwinner)
        ytrain_new = concate(ytrain, ywinner)


        index = 1
        print('training data concatenated')
        #while number_samples <= np.shape(Xunseen_orig)[0]:
        while np.shape(Xwinner)[0] > 0:
            #print(method.__name__,'at '+str(100*(np.shape(Xtrain_new)[0] - np.shape(Xtrain)[0])/np.shape(Xunseen_orig)[0])+' %')
            #just for debugging
            #if index == 1:
             #   model_new = retrain(model_base,epochs_retrain,mini_batch_size,Xtrain_new, ytrain_new)[0]
              #  print(tester(Xtest,ytest, model_new)[0])
            model_base = load_model(model_base_str)
            print('model loaded')
            model_new = retrain_early_stopping(model_base,epochs_retrain,mini_batch_size,Xtrain_new, ytrain_new, Xtest, ytest)[0]

            accuracy = tester(Xtest,ytest, model_new)[0]
            df.at[index, 'fraction used'] = round(1-np.shape(Xloser)[0]/np.shape(Xunseen_orig)[0],2)#number_samples
            df.at[index, str(method.__name__)] = accuracy
            print(df)
            Xwinner, ywinner, Xloser, yloser = seperation(Xloser, yloser, model_new, retrain_size, method)

            #Xtrain_new, ytrain_new = np.concatenate((Xtrain, Xwinner), axis=0), np.concatenate((ytrain, ywinner), axis=0)
            Xtrain_new, ytrain_new = concate(Xtrain, Xwinner), concate(ytrain, ywinner)

            index = index + 1

            del model_new, model_base



    df.to_csv('RESULTS.csv', index = False)
    print(df)
    return(df)


