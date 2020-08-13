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
#from tensorflow import keras
import keras.backend.tensorflow_backend as K
import tensorflow as tf

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

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


DATA_DIRECTORY = 'Data'

def savemodel(model,name):
    model.save_weights(str(name)+'.h5')
    model_json = model.to_json()
    with open(str(name)+'.json', "w") as json_file:
        json_file.write(model_json)
    json_file.close()
    print('model saved')

def loadmodel(name):
    from keras.models import model_from_json
    json_file = open(str(name)+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    model.load_weights(str(name)+'.h5')
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    print('model loaded')
    return(model)


def tester(Xtest,ytest,model):
    print('start evaluating')
    if type(model) == str:
        #K.clear_session()
        #model = tf.keras.models.load_model(model)
        model = loadmodel(model)

    loss = model.evaluate(Xtest, ytest, verbose=1)
    print('loss :'+ str(loss))

    ypred = model.predict(Xtest)

    ytest_flat = np.argmax(ytest, axis=1)
    ypred_flat = np.argmax(ypred, axis=1)

    print('acc :' + str(accuracy_score(ytest_flat, ypred_flat)))
    print(confusion_matrix(ytest_flat, ypred_flat))

    return(ypred)
