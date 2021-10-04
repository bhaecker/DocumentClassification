# -----------------------------------------------------------
# Backbone ml algorithms
# serving as/for
# Reinforcement oracles and Multi-armed bandits
# -----------------------------------------------------------

import sys
import numpy as np
import pickle
#from xgboost import XGBClassifier as XGB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, InputLayer, Input, Concatenate, Conv2D, Flatten, Dense

from .TransferLearning import fetch_data,retrain

def RL_model_dual(number_classes,output_dimension):
    '''
    set up dual (CNN and NN combined) oracle
    '''

    #CNN for image processing
    image_input = Input((244, 244, 3)) #same size as in CNN Model or numpy array of images
    conv_layer = Conv2D(32, (5, 5))(image_input)
    pool_layer = MaxPooling2D(pool_size=(5, 5))(conv_layer)
    dropout_layer = Dropout(0.5)(pool_layer)
    conv_layer = Conv2D(64, (5, 5))(dropout_layer)
    pool_layer = MaxPooling2D(pool_size=(3, 3))(conv_layer)
    dropout_layer = Dropout(0.5)(pool_layer)
    conv_layer = Conv2D(128, (3, 3))(dropout_layer)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)
    dropout_layer = Dropout(0.5)(pool_layer)
    conv_layer = Conv2D(256, (3, 3))(dropout_layer)

    flat_layer = Flatten()(conv_layer)

    #predictions from classification model
    prediction_input = Input((number_classes,))

    concat_layer = Concatenate()([prediction_input, flat_layer])
    dense_layer = Dense(256, activation="relu")(concat_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(128, activation="relu")(dropout_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(64, activation="relu")(dropout_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(output_dimension, activation="linear")(dropout_layer)#output the expected reward for decision "ask model" in first node and "ask human" in second node

    model = Model(inputs=[image_input, prediction_input], outputs=output_layer)

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])#IMPORTANT: loss function

    return(model)

def RL_model_mono(number_classes,output_dimension):
    '''
    set up mono (vanilla NN) oracle
    '''
    prediction_input = Input((number_classes,))
    dense_layer = Dense(256, activation="relu")(prediction_input)
    dropout_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(128, activation="relu")(dropout_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    dense_layer = Dense(64, activation="relu")(dropout_layer)
    dropout_layer = Dropout(0.5)(dense_layer)
    output_layer = Dense(output_dimension, activation="linear")(dropout_layer)  # output the expected reward for decision "ask model" in first node and "ask human" in second node

    model = Model(inputs=prediction_input, outputs=output_layer)

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])  # IMPORTANT: loss function

    return (model)

def RF_mono(number_trees):
    '''
    Random Forest Regressor
    '''
    oracle = RandomForestRegressor(n_estimators=number_trees, random_state=8)
    return(oracle)

def pretrain_dual_oracle(dual_oracle,CNN_model,epochs):
    '''
    pretrain an oracle (which sees images AND predictions) to predict increase of accuracy
    '''
    if type(CNN_model) == str:
        CNN_model = load_model(CNN_model)

    Xtrain, ytrain = fetch_data('trans_train')
    #Xtrain, ytrain = Xtrain[:50], ytrain[:50]
    Xtest, ytest = fetch_data('trans_test')
    #Xtest, ytest = Xtest[:50], ytest[:50]
    sample_size = np.shape(Xtrain)[0]
    reward = np.empty(sample_size)
    base_acc = CNN_model.evaluate(Xtest, ytest, verbose=0)[1]
    print('baseacc ' + str(base_acc))
    #get the rewards aka. improvements for all training data
    for idx in range(sample_size):
        print(idx / sample_size)
        #maybe retrain plus training samples?
        CNN_model_retrained = retrain(CNN_model,20,1,Xtrain[idx:idx+1],ytrain[idx:idx+1])[0]
        new_acc = CNN_model_retrained.evaluate(Xtest, ytest, verbose=0)[1]
        reward[idx] = new_acc - base_acc
    print(reward)

    # feed that into oracle
    ypred_train = CNN_model.predict(Xtrain)  # feed that into oracle
    oracle = dual_oracle
    oracle_name = "pretrained_dual_oracle.h5"

    try:
        oracle.fit(x=[Xtrain,ypred_train],y=reward,
                                validation_split=0,
                                batch_size=128,
                                epochs=epochs,
                                verbose=1)
        #oracle_name = "{}_{}_epochs.h5".format(oracle.__name__,epochs)
        oracle.save(oracle_name)
    except:
        oracle.fit(x=[Xtrain,ypred_train],y=reward)
        #oracle_name = "{}_{}.pkl".format(oracle.__name__,'prefitted')
        with open(oracle_name, 'wb') as file:
            pickle.dump(oracle, file)
    del Xtest, ytest, Xtrain, ytrain, ypred_train, reward
    return(oracle)


def pretrain_mono_oracle(mono_oracle,CNN_model,epochs):
    '''
    pretrain an oracle (which sees ONLY predictions) to predict increase of accuracy
    '''
    if type(CNN_model) == str:
        CNN_model = load_model(CNN_model)

    Xtrain, ytrain = fetch_data('trans_train')
    Xtrain, ytrain = Xtrain[:50], ytrain[:50]
    Xtest, ytest = fetch_data('trans_test')
    #Xtest, ytest = Xtest[:50], ytest[:50]
    sample_size = np.shape(Xtrain)[0]
    reward = np.empty(sample_size)
    base_acc = CNN_model.evaluate(Xtest, ytest, verbose=0)[1]
    print('baseacc ' + str(base_acc))
    for idx in range(sample_size):
        print(idx / sample_size)
        #maybe retrain plus training samples?
        CNN_model_retrained = retrain(CNN_model,20,1,Xtrain[idx:idx+1],ytrain[idx:idx+1])[0]
        new_acc = CNN_model_retrained.evaluate(Xtest, ytest, verbose=0)[1]
        reward[idx] = new_acc - base_acc
    print(reward)

    ypred_train = CNN_model.predict(Xtrain)  # feed that into oracle

    oracle = mono_oracle
    oracle_name = "pretrained_mono_oracle.h5"
    try:
        oracle.fit(x=ypred_train, y=reward,
                   validation_split=0,
                   batch_size=128,
                   epochs=epochs,
                   verbose=1)
        #oracle_name = "{}_{}_epochs.h5".format(oracle.__name__, epochs)
        oracle.save(oracle_name)
    except:
        oracle.fit(ypred_train, reward)
        oracle.fit(x=[Xtrain, ypred_train], y=reward)
        #oracle_name = "{}_{}.pkl".format(oracle.__name__, 'prefitted')
        with open(oracle_name, 'wb') as file:
            pickle.dump(oracle, file)

    del Xtest, ytest, Xtrain, ytrain, ypred_train, reward
    return(oracle)
