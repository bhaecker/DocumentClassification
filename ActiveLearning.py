from baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn
import sys
import math
import glob
import os
import numpy as np
import os.path as path
from scipy import misc
import imageio
import keras
from PIL import Image
import cv2
from tensorflow import keras

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

DATA_DIRECTORY = 'Data'

#unseen_data is a class in our setup
def seperation(model,unseen_data,batch_size,method):
    '''
    unseen data gets ranked with respect to the model predictions and method
    returns batch_sized batch of most informative samples and remaining samples
    '''

    # load unseen data
    R = np.load(DATA_DIRECTORY + '/Tobacco_unseen/'+unseen_data+'.npy')
    # transform blackWhite to RGB
    X = np.stack((R[:][:][:], R[:][:][:], R[:][:][:]), axis=3)
    #get the right data type for model prediction
    X = X.astype('float16')
    model = keras.models.load_model(model)
    ystar = model.predict(X)
    scores = np.array(list(map(method,ystar)))
    #todo check if we need the highest or lowest values for all functions
    n_highest = np.argpartition(scores, -batch_size)[-batch_size:]
    #TODO implement transforme blackWhite to RGB in preprocessing already !!
    #Todo mix trainingsdata etc

    #seperate unseen data in winner and looser data set
    Xwinner = X[n_highest,:,:]

    mask = np.ones_like(scores,dtype = bool)
    mask[n_highest] = False
    Xloser = X[mask,:,:]

    print(Xwinner.shape)
    print(Xloser.shape)

    return(Xwinner,Xloser)

print(seperation('model_0epochs.h5','Resume',3,margin_sampling_fn))