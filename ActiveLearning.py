# -----------------------------------------------------------
# Use Active Learning method to seperate training data
# -----------------------------------------------------------

import sys
import math
import glob
import os
import functools
import numpy as np
import collections

from tensorflow.keras.models import load_model

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

    if str(method.__name__).endswith('method') or str(method.__name__).endswith('algo'):
        Xwinner,ywinner,Xloser,yloser = method(X, y, batch_size, model)
        # get the class distribution
        class_distribution = collections.Counter(np.where(ywinner == 1)[1])
        print(class_distribution)
        return(Xwinner,ywinner,Xloser,yloser)

    if type(model) == str:
        model = load_model(model)
    ystar = model.predict(X)

    scores = np.array(list(map(method,ystar)))

    #todo check everytime, if we need the highest or lowest values for ALL functions
    #get the indices of the batch_sized highest scores
    n_highest = np.argpartition(scores, -batch_size)[-batch_size:]
    #print(n_highest)
    #seperate unseen data in winner and looser data set by the indices
    Xwinner = X[n_highest,:,:]
    ywinner = y[n_highest]

    # get the class distribution
    class_distribution = collections.Counter(np.where(ywinner == 1)[1])
    print(class_distribution)

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


