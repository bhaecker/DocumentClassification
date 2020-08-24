import sys
import math
import glob
import os
import numpy as np


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

#from .Testing import loadmodel,savemodel

DATA_DIRECTORY = 'DocumentClassification/Data/'
DATA_DIRECTORY = 'Data/' #TODO comment out when using server

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



def fetch_data(string):
    '''
    collect ALL training/test or unseen numpy arrays plus labels and shuffle them
    '''
    file = 'Tobacco_'+string+'/'
    labels = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']
    # load training data
    X = np.load(DATA_DIRECTORY + file + labels[0] + '.npy')
    # load labels
    r = np.load(DATA_DIRECTORY + file + labels[0] + '_target.npy')
    y = np.stack([r] * X.shape[0])
    for label in labels[1:]:
        # load training data
        Xstar = np.load(DATA_DIRECTORY + file + label + '.npy')
        X = np.concatenate((X,Xstar),axis=0)
        # load labels
        r = np.load(DATA_DIRECTORY + file + label + '_target.npy')
        ystar = np.stack([r] * Xstar.shape[0])
        y = np.concatenate((y,ystar),axis=0)


    s = np.arange(X.shape[0])
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    print(string+' data fetched')

    return(X,y)



def fine_tune(X,y,epochs,batch_size):
    '''
    fine tune Inception image network to X and y
    '''

    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # save model
    #base_model.save('base_model.h5')
    # load model
    #model = keras.models.load_model('model_1epochs.h5')
    print(base_model.summary())
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add fully-connected layers
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

    # train the model on the new data for a few epochs

    #print('training for class '+label)
    history_topDense = model.fit(X, y,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

    # at this point, the top layers are well trained and we can start fine-tuning
    # convolutional layers from inception V3. We will freeze the bottom N layers
    # and train the remaining top layers.

    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    #for i, layer in enumerate(model.layers):
        #print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    #for layer in model.layers[:249]:
        #layer.trainable = False
    for layer in model.layers:
        layer.trainable = True

    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy']) #todo Try increasing the batch size and reduce the learning rate while training.

    # we train our model again (this time fine-tuning the top 2 inception blocks
    # alongside the top Dense layers
    history_all = model.fit(X, y,
        validation_split = 0.2,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1)

    #model.save('model_'+str(epochs)+'epochs.h5')
    savemodel(model,'model_'+str(epochs)+'epochs')

    return(model,history_topDense,history_all)


def retrain(model,epochs,batch_size,X,y):
    '''
    retrain model
    '''
    if type(model) == str:
        model = loadmodel(model)
    print('start retraining')
    history = model.fit(X,y,
            #validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)
    savemodel(model,'retrained_'+str(epochs)+'epochs')
    return(model,history)

################
