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

from TransferLearning import fetch_data, fine_tune, retrain, savemodel, loadmodel
from Testing import tester
from ActiveLearning import seperation
#from baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn, mutural_info_uniform_fn, diff_uniform_fn
from MetricsMethod import metric_method, mutural_info_method,diversity_method
#from RandomForest import RandomForest_method

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, InputLayer, Input, Concatenate, Conv2D, Flatten, Dense

Xunseen, yunseen  = fetch_data('unseen')
Xunseen, yunseen = Xunseen[300:700], yunseen[300:700]
#ypred = tester(Xtest, ytest,model)[1]
model = loadmodel('model_40epochs')
for method in [metric_method]:
    Xtest, ytest, a,b = seperation(Xunseen, yunseen,model,50,method)
    print(str(method.__name__))
    ypred = tester(Xtest, ytest, model)[1]
    print(np.average(ypred,axis=0))



sys.exit()

number_features = 3
number_classes =3
epochs =3

input_layer = Input((number_features,))
x = Dense(16, activation='relu')(input_layer)
x = Dense(16, activation='relu')(input_layer)
predictions = Dense(number_classes, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=input_layer, outputs=predictions)
model.fit(Xtrain,ytrain,epochs=epochs)
model.evaluate(Xtest,ytest)

for epoch in range(epochs):
    model.fit(Xtrain,ytrain,epochs=epoch)



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

