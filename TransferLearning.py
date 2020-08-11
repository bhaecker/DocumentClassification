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

def fine_tune():
    # create the base pre-trained model
    base_model = InceptionV3(weights='imagenet', include_top=False)
    # save model
    #base_model.save('base_model.h5')
    # load model
    #model = keras.models.load_model('model_1epochs.h5')
    #print(base_model.summary())
    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 10 classes
    predictions = Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # compile the model (should be done *after* setting layers to non-trainable)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

    #train each class
    labels = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']
    for label in labels:
        #load training data
        R = np.load(DATA_DIRECTORY+'/Tobacco_train/'+label+'.npy')
        #tranform blackWhite to RGB
        X = np.stack((R[:][:][:],R[:][:][:],R[:][:][:]), axis=3)
        #load labels
        r = np.load(DATA_DIRECTORY+'/Tobacco_train/'+label+'_target.npy')
        y = np.stack([r]*X.shape[0])


        # train the model on the new data for a few epochs
        batch_size = 32
        epochs = 1
        print('training for class '+label)
        model.fit(X, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        #for i, layer in enumerate(base_model.layers):
            #print(i, layer.name)

        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from tensorflow.keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        model.fit(X, y,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)

        model.save('model_'+str(epochs)+'epoch.h5')

fine_tune()
