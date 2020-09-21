import sys
import numpy as np

from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, InputLayer, Input, Concatenate, Conv2D, Flatten, Dense

from .TransferLearning import fetch_data
from .Backbone import RL_model_mono, RL_model_dual

##TODO: reinforcemnt learning with explore then exploit and Softmaxexplorer

def epsgreedy_dual_oracle(Xtrain,ytrain,RL_dual,CNN_model,num_episodes):
    '''
    a correct prediction is

    '''

    #just for testing:
    Xtest,ytest = fetch_data('test')
    y_pred_test = CNN_model.predict(Xtest)

    RL_model = RL_dual

    #y = 0.95
    eps = 0.5
    decay_factor = 0.999
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
                r -= 0.1
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
                            epochs=10,
                            verbose=0)
            idx += 1
            r_sum += r

        RL_model.save('RL_model_dual_'+str(i)+'.h5')
        r_avg_list.append(r_sum / 1000)

        #just for testing

        expected_rewards = RL_model.predict([Xtest, y_pred_test])
        print(expected_rewards)


    print(r_avg_list)

    return(RL_model)


def epsgreedy_mono_oracle(Xtrain,ytrain,RL_mono,CNN_model,num_episodes):
    '''
    train the RL model, which gets as inputs the images and the predictions of the images
    based on a policy, the RL model learns to predict its reward, when choosing either the CNN model to predict or an orcale (human annotator)
    '''

    #just for testing:
    Xtest,ytest = fetch_data('test')
    y_pred_test = CNN_model.predict(Xtest)

    RL_model = RL_mono

    #y = 0.95
    eps = 0.5
    decay_factor = 0.999
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
                decision = np.argmax(RL_model.predict(predicted_class))
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
                r -= 0.1
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
            target_vec = RL_model.predict(predicted_class)[0]
            #print(target_vec)
            target_vec[decision] = target
            #print(target_vec)

            RL_model.fit(x=predicted_class,y=target_vec.reshape(-1, 2),#
                            validation_split=0,
                            batch_size=1,
                            epochs=10,
                            verbose=0)
            idx += 1
            r_sum += r

        RL_model.save('RL_model_mono_'+str(i)+'.h5')
        r_avg_list.append(r_sum / 1000)

        #just for testing

        expected_rewards = RL_model.predict([Xtest, y_pred_test])
        print(expected_rewards)


    print(r_avg_list)

    return(RL_model)

def RL_human_method(X, y, batch_size, CNN_model):
    '''

    '''
    number_samples = np.shape(X)[0]
    if number_samples <= batch_size:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return(X,y,X_empty,y_empty)

    if type(CNN_model) == str:
        CNN_model = load_model(CNN_model)

    y_pred = CNN_model.predict(X)

    RL_model = load_model('RL_model_4.h5')

    try:
        expected_rewards = RL_model.predict([X,y_pred])
    except:
        expected_rewards = RL_model.predict(y_pred)

    #decisions = [np.random.choice(2, p = expected_rewards[i]) for i in range(number_samples)]

    #reminder: first entry expected reward for asking CNN model, second entry for asking human
    ##either sort for lowest expected reward for CNN model, or highest expected reward for asking human
    sort_ind = np.argsort(expected_rewards[:,1])
    #sort samples (and labels) in descending order from highest expected reward to lowest
    X = X[sort_ind[::-1]]
    y = y[sort_ind[::-1]]

    # just for testing
    #y_pred = y_pred[sort_ind[::-1]]
    #print(RL_model.predict([X, y_pred]))

    Xwinner, ywinner = X[:batch_size], y[:batch_size]
    Xloser, yloser = X[batch_size:], y[batch_size:]

    return(Xwinner, ywinner, Xloser, yloser)


def RL_CNN_method(X, y, batch_size, CNN_model):
    '''

    '''
    number_samples = np.shape(X)[0]
    if number_samples <= batch_size:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return (X, y, X_empty, y_empty)

    if type(CNN_model) == str:
        CNN_model = load_model(CNN_model)

    y_pred = CNN_model.predict(X)

    RL_model = load_model('RL_model_4.h5')

    try:
        expected_rewards = RL_model.predict([X, y_pred])
    except:
        expected_rewards = RL_model.predict(y_pred)

    # decisions = [np.random.choice(2, p = expected_rewards[i]) for i in range(number_samples)]

    # reminder: first entry expected reward for asking CNN model, second entry for asking human
    ##either sort for lowest expected reward for CNN model, or highest expected reward for asking human
    sort_ind = np.argsort(expected_rewards[:, 0])
    # sort samples (and labels) in ascending order from lowest expected reward to highest
    X = X[sort_ind]
    y = y[sort_ind]


    #just for testing
    #y_pred = y_pred[sort_ind]
    #print(RL_model.predict([X, y_pred]))


    Xwinner, ywinner = X[:batch_size], y[:batch_size]
    Xloser, yloser = X[batch_size:], y[batch_size:]

    print(np.shape(Xwinner))
    print(np.shape(Xloser))
    return (Xwinner, ywinner, Xloser, yloser)



