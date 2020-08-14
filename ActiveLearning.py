import sys
import math
import glob
import os
import numpy as np
import keras
import tensorflow as tf

#config = tf.config.experimental(device_count = {'GPU': 1 , 'CPU': 1} )
#sess = tf.Session(config=config)
#keras.backend.set_session(sess)

#from tensorflow import keras

DATA_DIRECTORY = 'Data'

#unseen_data is a class in our setup
def seperation(X,y,model,batch_size,method):
    '''
    unseen data gets ranked with respect to the model predictions and method
    returns batch_sized batch of most informative samples and remaining samples
    '''
    try:
        ystar = model.predict(X)
    except:
        model = keras.models.load_model(model)
        ystar = model.predict(X)
    #print('happy')
    scores = np.array(list(map(method,ystar)))
    #todo check if we need the highest or lowest values for all functions
    n_highest = np.argpartition(scores, -batch_size)[-batch_size:]

    #seperate unseen data in winner and looser data set
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

