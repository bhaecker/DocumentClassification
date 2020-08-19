import sys
import math
import glob
import os
import numpy as np
import pandas as pd


#from tensorflow import keras
#import keras.backend.tensorflow_backend as K
import keras
import tensorflow as tf

from keras.models import model_from_json

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from .TransferLearning import fetch_data, fine_tune, retrain, savemodel, loadmodel
from .ActiveLearning import seperation
from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn



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

    #todo: retrain with new and old samples

    if type(model_base) == str:
        model_base = loadmodel(model_base)

    Xtest, ytest = fetch_data('test')
    #Xtest, ytest = Xtest[:10], ytest[:10]

    base_performance = tester(Xtest, ytest, model_base)[0]
    number_samples = 0
    df = pd.DataFrame([[number_samples]+[base_performance]*len(list_methods)], columns = ['number of samples'] + [method.__name__ for method in list_methods])

    Xunseen_orig, yunseen_orig = fetch_data('unseen')
    print(np.shape(Xunseen_orig))
    #Xunseen_orig, yunseen_orig = Xunseen_orig[:10], yunseen_orig[:10]

    for method in list_methods:

        Xwinner, ywinner, Xloser, yloser = seperation(Xunseen_orig, yunseen_orig, model_base, retrain_size, method)
        number_samples = retrain_size
        model_old = model_base
        index = 1
        while number_samples < np.shape(Xunseen_orig)[0]:
            print(method.__name__,number_samples)
            model_new = retrain(model_old,epochs_retrain,mini_batch_size,Xwinner,ywinner)[0]
            accuracy = tester(Xtest,ytest, model_new)[0]
            df.at[index, 'number of samples'] = number_samples
            df.at[index, str(method.__name__)] = accuracy

            Xwinner, ywinner, Xloser, yloser = seperation(Xloser, yloser, model_new, retrain_size, method)

            #Xunseen, yunseen = Xloser, yloser
            index = index + 1
            print(df)
            #df.loc[len(df)] = [number_samples] + accuracy_list
            number_samples = number_samples + retrain_size
            model_old = model_new

    return(df)