import sys
import math
import glob
import os
import numpy as np
import pandas as pd
import collections


#from tensorflow import keras
#import keras.backend.tensorflow_backend as K
import keras
import tensorflow as tf

from keras.models import model_from_json

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from TransferLearning import fetch_data, fine_tune, retrain, savemodel, loadmodel
from Testing import tester
from ActiveLearning import seperation
from baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn
from MetricsMethod import metric_method

model = loadmodel('model_40epochs')

Xunseen, yunseen  = fetch_data('unseen')
Xunseen, yunseen = Xunseen[1:200], yunseen[1:200]
#ypred = tester(Xtest, ytest,model)[1]
for method in [metric_method,entropy_fn, least_confident_fn, margin_sampling_fn, random_fn]:
    Xtest, ytest, a,b = seperation(Xunseen, yunseen,model,50,method)
    print(str(method.__name__))
    ypred = tester(Xtest, ytest, model)[1]
    print(np.average(ypred,axis=0))

##=> random chosen examples have a much higher accuracy, so they should be less usefull for retraining
##=> since 'seperation' function was used, this should be bug free

