# -----------------------------------------------------------
# Contextual Multi-armed Bandits for AL
# -----------------------------------------------------------

import sys
import random
import numpy as np
import pickle
#from xgboost import XGBClassifier as XGB
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from tensorflow.keras.models import load_model
from .TransferLearning import fetch_data,retrain

#from .Backbone import RL_model_mono, RL_model_dual, pretrain_dual_oracle, pretrain_mono_oracle


def ContextualAdaptiveGreedy_mono_algo(Xunseen, yunseen, batch_size, CNN_model):
    '''
    uses an oracle (which ONLY sees predictions) to predict the improvement of the CNN model
    the oracle sees the predictions of the CNN model
    when it chooses a sample, the true improvement is revealed to the oracle
    '''
    threshold = 0.5
    decay_rate = 0.99
    number_rounds = 2
    offline_batchsize = 100
    epochs = 10


    #oracle = pretrain_oracle(CNN_model) most likely unnecessary
    #oracle = LogisticRegression() this is a classifier
    #oracle = RandomForestClassifier() classifier dont work since we have the continous reward threshold

    #try:
     #   pkl_filename = "contextual_oracle.pkl"
      #  with open(pkl_filename, 'rb') as file:
       #     oracle = pickle.load(file)
    #except:
     #   oracle = RandomForestRegressor(n_estimators=500, random_state=8)

    oracle = load_model('RL_model_mono_100_epochs.h5')

    Xtest, ytest = fetch_data('test')
    #Xtest, ytest = Xtest[:10], ytest[:10]
    base_acc = CNN_model.evaluate(Xtest, ytest, verbose=0)[1]

    number_samples = np.shape(Xunseen)[0]
    print(number_samples)
    if number_samples <= batch_size:
        X_empty = np.empty([0, np.shape(Xunseen)[1], np.shape(Xunseen)[2], np.shape(Xunseen)[3]])
        y_empty = np.empty([0, np.shape(yunseen)[1]])
        return(Xunseen, yunseen, X_empty, y_empty)

    if type(CNN_model) == str:
        CNN_model = load_model(CNN_model)

    #this is our context
    ypred_unseen = CNN_model.predict(Xunseen)
    ypred_unseen = np.array(ypred_unseen)

    for round in range(number_rounds):
        print('round number: '+str(round))
        winner_idx_list = []
        rewards = []
        for sample in range(offline_batchsize):
            #let the oracle predict the reward for each element of the context
            try:
                expected_reward = oracle.predict(ypred_unseen)
                #print('tried successful')
            except:
                expected_reward = 0

            #if there is a reward which is higher then the threshold, chose the corresponding sample, if not choose random
            if np.max(expected_reward) > threshold:
                winner_idx = np.argmax(expected_reward)
                #print('winner '+str(winner_idx))
            else:
                winner_idx = random.sample(range(number_samples),1)[0]
                #print('random ' + str(winner_idx))
            #decrease threshold
            winner_idx_list.append(winner_idx)
            threshold = threshold*decay_rate
            #threshold = threshold - decay_rate#change later

            #reveal the real reward for the choosen context, aka. label the sample, retrain the CNN model and calculate delta accuracy
            CNN_model_retrained = retrain(CNN_model, 1, 1, Xunseen[winner_idx:winner_idx + 1], yunseen[winner_idx:winner_idx + 1])[0]
            new_acc = CNN_model_retrained.evaluate(Xtest, ytest, verbose=0)[1]
            reward = new_acc - base_acc
            rewards.append(reward)

        #retrain the oracle with the choosen sample and the real reward
        try:
            oracle.fit(x=ypred_unseen[winner_idx_list], y=np.array(rewards),
                       validation_split=0,
                       batch_size=1,
                       epochs=epochs,
                       verbose=0)
        except:
            oracle.fit(ypred_unseen[winner_idx_list], rewards)

    #use oracle to predict rewards aka. improvement of training process
    expected_reward = oracle.predict(ypred_unseen)
    n_farest =  np.argpartition(expected_reward, -batch_size)[-batch_size:]

    #seperate unseen data in winner and looser data set by the indices
    Xwinner = Xunseen[n_farest, :, :]
    ywinner = yunseen[n_farest]

    mask = np.ones(Xunseen.shape[0], dtype=bool)
    mask[n_farest] = False
    Xloser = Xunseen[mask, :, :]
    yloser = yunseen[mask]

    return(Xwinner, ywinner, Xloser, yloser)



def ContextualAdaptiveGreedy_dual_algo(Xunseen, yunseen, batch_size, CNN_model):
    '''
    uses an oracle (which sees images AND predictions) to predict the improvement of the CNN model
    the oracle sees the predictions of the CNN model
    when it chooses a sample, the true improvement is revealed to the oracle
    '''
    threshold = 0.5
    decay_rate = 0.99
    number_rounds = 2
    offline_batchsize = 10
    epochs = 10

    number_samples = np.shape(Xunseen)[0]
    if number_samples <= batch_size:
        X_empty = np.empty([0, np.shape(Xunseen)[1], np.shape(Xunseen)[2], np.shape(Xunseen)[3]])
        y_empty = np.empty([0, np.shape(yunseen)[1]])
        return(Xunseen, yunseen, X_empty, y_empty)

    if type(CNN_model) == str:
        CNN_model = load_model(CNN_model)

    oracle = load_model('RL_model_dual_100_epochs.h5')

    Xtest, ytest = fetch_data('test')
    # Xtest, ytest = Xtest[:10], ytest[:10]
    base_acc = CNN_model.evaluate(Xtest, ytest, verbose=0)[1]

    #this plus Xunseen is our context
    ypred_unseen = CNN_model.predict(Xunseen)
    ypred_unseen = np.array(ypred_unseen)

    for round in range(1,number_rounds):
        print('round number: '+str(round))
        winner_idx_list = []
        rewards = []
        for sample in range(offline_batchsize):
            #let the oracle predict the reward for each element of the context
            try:
                expected_reward = oracle.predict([Xunseen,ypred_unseen])
                #print('tried successful')
            except:
                expected_reward = 0

            #if there is a reward which is higher then the threshold, chose the corresponding sample, if not choose random
            if np.max(expected_reward) > threshold:
                winner_idx = np.argmax(expected_reward)
                #print('winner '+str(winner_idx))
            else:
                winner_idx = random.sample(range(number_samples),1)[0]
                #print('random ' + str(winner_idx))
            #decrease threshold
            winner_idx_list.append(winner_idx)
            threshold = threshold*decay_rate
            #threshold = threshold - decay_rate#change later

            #reveal the real reward for the choosen context, aka. label the sample, retrain the CNN model and calculate delta accuracy
            CNN_model_retrained = retrain(CNN_model, 1, 1, Xunseen[winner_idx:winner_idx + 1], yunseen[winner_idx:winner_idx + 1])[0]
            new_acc = CNN_model_retrained.evaluate(Xtest, ytest, verbose=0)[1]
            reward = new_acc - base_acc
            rewards.append(reward)

        #retrain the oracle with the choosen sample and the real reward
        try:
            oracle.fit(x=[Xunseen[winner_idx_list], ypred_unseen[winner_idx_list]], y=np.array(rewards),
                        validation_split=0,
                        batch_size=1,
                        epochs=epochs,
                        verbose=0)
        except:
            oracle.fit(x=[Xunseen[winner_idx_list], ypred_unseen[winner_idx_list]], y=np.array(rewards))

    #use oracle to predict rewards aka. improvement of training process
    expected_rewards = oracle.predict([Xunseen,ypred_unseen])
    #flatten the list
    expected_rewards = [expected_reward[0] for expected_reward in expected_rewards]

    n_farest =  np.argpartition(expected_rewards, -batch_size)[-batch_size:]

    #seperate unseen data in winner and looser data set by the indices
    Xwinner = Xunseen[n_farest, :, :]
    ywinner = yunseen[n_farest]

    mask = np.ones(Xunseen.shape[0], dtype=bool)
    mask[n_farest] = False
    Xloser = Xunseen[mask, :, :]
    yloser = yunseen[mask]

    return(Xwinner, ywinner, Xloser, yloser)





