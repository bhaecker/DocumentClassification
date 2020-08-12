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

# load model
model = keras.models.load_model('model_1epochs.h5')
labels = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']
for label in labels:
    # load testing data
    R = np.load(DATA_DIRECTORY + '/Tobacco_test/' + label + '.npy')
    # transform blackWhite to RGB

    X = np.stack((R[:][:][:], R[:][:][:], R[:][:][:]), axis=3)
    # load labels
    r = np.load(DATA_DIRECTORY + '/Tobacco_test/' + label + '_target.npy')
    y = np.stack([r] * X.shape[0])

    stats = model.evaluate(X, y,  verbose = 0 )#, batch_size=50)
    print('loss for class '+ label + ': ' + str(stats))