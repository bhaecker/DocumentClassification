import sys
import math
import glob
import os
import numpy as np
import tensorflow as tf
#from tensorflow import keras

#config = tf.config.experimental(device_count = {'GPU': 1 , 'CPU': 1} )
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)

from .MetricsMethod import metric_method, mutural_info_method
from .RandomForest import RandomForest_method
from .TransferLearning import loadmodel

DATA_DIRECTORY = 'Data'

def seperation(X,y,model,batch_size,method):
    '''
    unseen data gets ranked with respect to the model predictions and method.
    returns batch_sized batch of most informative samples and remaining samples.
    '''
    if np.shape(X)[0] == 0:
        return(X,y,X,y)
    if batch_size > np.shape(X)[0]:
        X_empty = np.empty([0,np.shape(X)[1],np.shape(X)[2],np.shape(X)[3]])
        y_empty = np.empty([0,np.shape(y)[1]])
        return(X,y,X_empty,y_empty)

    if method == mutural_info_method:
        Xwinner, ywinner, Xloser, yloser = mutural_info_method(X,y,batch_size,model)
        return(Xwinner, ywinner, Xloser, yloser)

    if method == metric_method:
        Xwinner, ywinner, Xloser, yloser = metric_method(X,y,batch_size,model)
        return(Xwinner, ywinner, Xloser, yloser)

    if model == RandomForest_method:
        Xwinner, ywinner, Xloser, yloser = RandomForest_method(X, y, batch_size, model)
        return (Xwinner, ywinner, Xloser, yloser)


    if type(model) == str:
        model = loadmodel(model)
    ystar = model.predict(X)

    scores = np.array(list(map(method,ystar)))

    #todo check everytime, if we need the highest or lowest values for ALL functions
    #get the indices of the batch_sized highest scores
    n_highest = np.argpartition(scores, -batch_size)[-batch_size:]

    #seperate unseen data in winner and looser data set by the indices
    Xwinner = X[n_highest,:,:]
    ywinner = y[n_highest]

    mask = np.ones_like(scores,dtype = bool)
    mask[n_highest] = False
    Xloser = X[mask,:,:]
    yloser = y[mask]

    print('seperation made')

    #print(Xwinner.shape)
    #print(Xloser.shape)
    #print(ywinner.shape)
    #print(yloser.shape)
    return(Xwinner,ywinner,Xloser,yloser)


