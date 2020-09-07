from __future__ import print_function
import numpy as np

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

from TransferLearning import fetch_data, loadmodel

def data():
    x_unseen, y_unseen = fetch_data('unseen')
    x_test, y_test = fetch_data('test')
    CNN_model = loadmodel('model_40epochs')
    y_unseen_predicted = CNN_model.predict(x_unseen)
    y_test_predicted = CNN_model.predict(x_test)

    target_unseen = np.empty(np.shape(x_unseen)[0],2)
    target_test = np.empty(np.shape(x_test)[0], 2)

    r = 0
    if decision == 0:
        # CNN predicts correct
        if np.argmax(target_class, axis=1) == np.argmax(predicted_class, axis=1):
            r += 1
            # print('CNN predicted correct')
        # CNN predicts incorrect
        else:
            # print('CNN predicted INcorrect')
            r -= 1.1
    # RL model decides for human prediction
    else:
        # print('please help me human')
        r -= 0.05

    return([x_unseen,y_unseen_predicted], y_unseen, [x_test,y_test_predicted], y_test)