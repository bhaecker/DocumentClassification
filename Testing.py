import sys
import math
import glob
import os
import numpy as np



#from tensorflow import keras
#import keras.backend.tensorflow_backend as K
import keras
import tensorflow as tf

from keras.models import model_from_json

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

    loss = model.evaluate(Xtest, ytest, verbose=0)
    print('loss :'+ str(loss))

    ypred = model.predict(Xtest)

    ytest_flat = np.argmax(ytest, axis=1)
    ypred_flat = np.argmax(ypred, axis=1)

    print('acc :' + str(accuracy_score(ytest_flat, ypred_flat)))
    print(confusion_matrix(ytest_flat, ypred_flat))

    return(ypred)
