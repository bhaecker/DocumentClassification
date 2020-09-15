import sys
import math
import glob
import os
import numpy as np
import pandas as pd
import collections

import tensorflow as tf

from tensorflow.keras.models import model_from_json

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#from TransferLearning import fetch_data, fine_tune, retrain, savemodel, loadmodel
#from Testing import tester
#from ActiveLearning import seperation
#from baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn, mutural_info_uniform_fn, diff_uniform_fn
#from MetricsMethod import metric_method, mutural_info_method,diversity_method
#from RandomForest import RandomForest_method

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, InputLayer, Input, Concatenate, Conv2D, Flatten, Dense
from tensorflow.keras.models import load_model

from tensorflow.keras.backend import manual_variable_initialization

'''Test how saving and loading of a trained model influences the accuracy.
Options are: compile after loading or not.
'''

import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model

#set seeds for reproducability
np.random.seed(42)
tf.random.set_seed(42)

#Set to true or false and observe different outcomes
compile_after_loading = False

#parameters for NN and data set
number_samples = 5000
number_features = 10
number_classes = 2
split_size = 0.8
epochs = 80

#random numbers as data set
data_set = np.random.random((number_samples,number_features+1))

#last column is target (binary for cases when the sum of all features is less or more then 5)
mask  = np.sum(data_set[:,:-1],axis=1) < 5
data_set[:,-1] = mask.astype(int)

#split in training and test set
Xtrain,ytrain = data_set[:math.floor(number_samples*split_size),:-1],data_set[:math.floor(number_samples*split_size),-1]
Xtest,ytest = data_set[math.floor(number_samples*split_size):,:-1],data_set[math.floor(number_samples*split_size):,-1]

#create a small vanilla keras model, which is always initialized with the same weights
def create_model(number_features,number_classes):
    input_layer = Input((number_features,))
    initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1., seed=42)
    x = Dense(16, activation='relu',kernel_initializer=initializer)(input_layer)
    x = Dense(16, activation='relu',kernel_initializer=initializer)(x)
    predictions = Dense(number_classes, activation='sigmoid',kernel_initializer=initializer)(x)
    model = Model(inputs=input_layer,outputs=predictions)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return(model)

#train in one session
model = create_model(number_features,number_classes)
model.fit(Xtrain,ytrain,epochs=epochs,verbose=0)
model.save('throwawaymodel.h5')
#score = model.evaluate(Xtest,ytest,verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
del model
model = load_model('throwawaymodel.h5')
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = model.evaluate(Xtest,ytest,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
sys.exit()
#train in different sessions
model = create_model(number_features,number_classes)
#save model after each epoch, load the model and retrain one epoch
for epoch in range(epochs):
    model.fit(Xtrain,ytrain,epochs=1,verbose=0)
    model.save('throwawaymodel.h5')
    del model
    model = load_model('throwawaymodel.h5')
    if compile_after_loading:
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#observe how accuracy matches the accuracy of the model trained in one session
#or drops, when compiling after each load_model
score = model.evaluate(Xtest,ytest,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])



sys.exit()

#new base model
model = create_model(number_features,number_classes)
model.fit(Xtrain,ytrain,epochs=epochs-epochsleft,verbose=0)
print(model.evaluate(Xtest,ytest))
model.save('throwawaymodel')
del model
model = load_model('throwawaymodel')
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Xtrain,ytrain,epochs=epochsleft,verbose=0)
print(model.evaluate(Xtest,ytest))

#model.save('throwawaymodel')
    #savemodel(model,'throwawaymodel')
    #with open(pkl_filename, 'wb') as file:
     #   pickle.dump(model, file)
    #model = load_model('throwawaymodel')
    #model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #model = loadmodel('throwawaymodel')
    #with open(pkl_filename, 'wb') as file:
    #pickle.dump(model, file)


#100 epochs
[2.2274699096679687, 0.459]
[0.38891122579574583, 0.615]
[0.3921713047027588, 0.628]

#epochs = 500
[2.422634494781494, 0.447]
[0.05634847607203119, 0.984]
[0.39176710438728335, 0.879]
#######

sys.exit()
model = loadmodel('model_40epochs')

Xunseen, yunseen  = fetch_data('unseen')
Xunseen, yunseen = Xunseen[300:700], yunseen[300:700]
#ypred = tester(Xtest, ytest,model)[1]
for method in [diversity_method,least_confident_fn,metric_method,diversity_method, random_fn, diff_uniform_fn, mutural_info_method]:
    Xtest, ytest, a,b = seperation(Xunseen, yunseen,model,50,method)
    print(str(method.__name__))
    ypred = tester(Xtest, ytest, model)[1]
    print(np.average(ypred,axis=0))

##=> random chosen examples have a much higher accuracy, so they should be less usefull for retraining
##=> since 'seperation' function was used, this should be bug free
##=> metric_method has very high accuracy, diff_uniform_fn very low

[0.051768106736475604, 0.977]