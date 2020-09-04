import sys
import numpy as np

from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, InputLayer, Input, Concatenate, Conv2D, Flatten, Dense

from tensorflow.keras.utils import plot_model

from .TransferLearning import fetch_data, loadmodel, savemodel


def RL_model(number_classes):

    #CNN for image processing
    image_input = Input((244, 244, 3)) #same size as in CNN Model or numpy array of images
    conv_layer = Conv2D(16, (7, 7))(image_input)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)
    conv_layer = Conv2D(32, (5, 5))(pool_layer)
    pool_layer = MaxPooling2D(pool_size=(3, 3))(conv_layer)
    conv_layer = Conv2D(64, (3, 3))(pool_layer)
    pool_layer = MaxPooling2D(pool_size=(5, 5))(conv_layer)
    conv_layer = Conv2D(64, (3, 3))(pool_layer)
    flat_layer = Flatten()(conv_layer)

    #predictions from classification model
    prediction_input = Input((number_classes,))

    concat_layer = Concatenate()([prediction_input, flat_layer])
    dense_layer = Dense(256, activation="relu")(concat_layer)
    dropout_layer = Dropout(0.3)(dense_layer)
    dense_layer = Dense(256, activation="relu")(dropout_layer)
    #dropout_layer = Dropout(0.3)(dense_layer)
    #dense_layer = Dense(256, activation="relu")(dropout_layer)
    dropout_layer = Dropout(0.3)(dense_layer)
    output_layer = Dense(2, activation="linear")(dropout_layer)#output the expected reward for decision "ask model" first node and "ask human" second node

    model = Model(inputs=[image_input, prediction_input], outputs=output_layer)

    model.compile(loss='mse', optimizer='adam', metrics=['mae'])#IMPORTANT: loss function

    return(model)


def train_RL_model(Xtrain,ytrain,RL_model,CNN_model,num_episodes):
    #y = 0.95
    eps = 0.5
    decay_factor = 0.99
    r_avg_list = []
    for i in range(num_episodes):
        print('episode:',i)
        #exploration to explotation
        eps *= decay_factor
        r_sum = 0
        number_sample = np.shape(Xtrain)[0]
        idx = 0
        while idx < number_sample-1:
            r = 0
            #prepare input for RL model
            sample = Xtrain[idx]
            sample = np.expand_dims(sample, axis=0)
            target_class = ytrain[idx]
            target_class = np.expand_dims(target_class, axis=0)

            predicted_class = CNN_model.predict(sample)
            #choose random action or let RL model decide
            if np.random.random() < eps:
                decision = np.random.randint(0, 2)
            else:
                decision = np.argmax(RL_model.predict([sample,predicted_class]))
            #consequence of decision
            #RL model decides for CNN prediction
            if decision == 0:
                #CNN predicts correct
                if np.argmax(target_class, axis=1) == np.argmax(predicted_class, axis=1):
                    r += 1
                    #print('CNN predicted correct')
                # CNN predicts incorrect
                else:
                    #print('CNN predicted INcorrect')
                    r -= 1
            #RL model decides for human prediction
            else:
                #print('please help me human')
                r -= 0.05
            #next step in environment
            #sample_new = Xtrain[idx+1]
            #sample_new = np.expand_dims(sample_new, axis=0)
            #predicted_class_new = CNN_model.predict(sample_new)
            #
            #weird = RL_model.predict([sample_new, predicted_class_new])
            #print(weird)
            #print(np.max(weird))
            target = r #+ y * np.max(weird)
            #reniforce the decision
            target_vec = RL_model.predict([sample,predicted_class])[0]
            #print(target_vec)
            target_vec[decision] = target
            #print(target_vec)

            RL_model.fit(x=[sample,predicted_class],y=target_vec.reshape(-1, 2),#
                            validation_split=0,
                            batch_size=1,
                            epochs=1,
                            verbose=0)
            idx += 1
            r_sum += r

        r_avg_list.append(r_sum / 1000)

    print(r_avg_list)
    savemodel(RL_model,'Rl_model')
    return(RL_model)

#TODO try out different output layer, float between 0 and 1
#more episodes
#try only with one input type...