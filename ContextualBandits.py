import sys
import random
import numpy as np
import pickle
#from xgboost import XGBClassifier as XGB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .TransferLearning import fetch_data,loadmodel,retrain

def pretrain_oracle(CNN_model):
    if type(CNN_model) == str:
        CNN_model = loadmodel(CNN_model)

    Xtrain, ytrain = fetch_data('train')
    Xtrain, ytrain = Xtrain[:50], ytrain[:50]
    Xtest, ytest = fetch_data('test')
    #Xtest, ytest = Xtest[:50], ytest[:50]
    sample_size = np.shape(Xtrain)[0]
    reward = np.empty(sample_size)
    base_acc = CNN_model.evaluate(Xtest, ytest, verbose=0)[1]
    print('baseacc ' + str(base_acc))
    for idx in range(sample_size):
        print(idx / sample_size)
        #maybe retrain plus training samples?
        CNN_model_retrained = retrain(CNN_model,100,1,Xtrain[idx:idx+1],ytrain[idx:idx+1])[0]
        new_acc = CNN_model_retrained.evaluate(Xtest, ytest, verbose=0)[1]
        reward[idx] = new_acc - base_acc
    print(reward)

    ypred_train = CNN_model.predict(Xtrain)  # feed that into oracle
    oracle = RandomForestRegressor(n_estimators=100, random_state=8)
    oracle.fit(ypred_train, reward)
    del Xtest, ytest, Xtrain, ytrain, ypred_train, reward

    return(oracle)


def ContextualAdaptiveGreedy(Xunseen, yunseen, batch_size, CNN_model, oracle):
    '''

    '''
    threshold = 0.5
    decay_rate = 0.99
    number_rounds = 3
    offline_batchsize = 100

    #oracle = pretrain_oracle(CNN_model)
    #oracle = LogisticRegression() this is a classifier
    #oracle = RandomForestClassifier() classifier doesnt work since we have the continous reward threshold
    oracle = RandomForestRegressor(n_estimators=321, random_state=8)

    Xtest, ytest = fetch_data('test')
    #Xtest, ytest = Xtest[:10], ytest[:10]
    base_acc = CNN_model.evaluate(Xtest, ytest, verbose=0)[1]

    # with open('RF', 'rb') as f:
    #   oracle = pickle.load(f)
    print('baseacc '+str(base_acc))

    number_samples = np.shape(Xunseen)[0]
    if number_samples <= batch_size:
        X_empty = np.empty([0, np.shape(Xunseen)[1], np.shape(Xunseen)[2], np.shape(Xunseen)[3]])
        y_empty = np.empty([0, np.shape(yunseen)[1]])
        return (Xunseen, Xunseen, X_empty, y_empty)

    if type(CNN_model) == str:
        CNN_model = loadmodel(CNN_model)

    #this is our context
    print(np.shape(Xunseen))
    ypred_unseen = CNN_model.predict(Xunseen)
    ypred_unseen = np.array(ypred_unseen)
    print(ypred_unseen)

    #print(ypred_unseen)

    for round in range(number_rounds):
        winner_idx_list = []
        rewards = []
        for sample in range(offline_batchsize):
        #let the oracle predict the reward for each element of the context
            try:
                expected_reward = oracle.predict(ypred_unseen)
            except:
                expected_reward = 0

            print('expected_reward:')
            print(expected_reward)

            #if there is a reward which is higher then the threshold, chose the corresponding sample, if not choose random
            if np.max(expected_reward) > threshold:
                winner_idx = np.argmax(expected_reward)
                print('winner '+str(winner_idx))
            else:
                winner_idx = random.sample(range(number_samples),1)[0]
                print('random ' + str(winner_idx))
            #decrease threshold
            winner_idx_list.append(winner_idx)
            threshold = threshold*decay_rate
            #threshold = threshold - decay_rate#change later
            print(threshold)
            #reveal the real reward for the choosen context, aka. label the sample, retrain the CNN model and calculate delta accuracy
            CNN_model_retrained = retrain(CNN_model, 10, 1, Xunseen[winner_idx:winner_idx + 1], yunseen[winner_idx:winner_idx + 1])[0]
            new_acc = CNN_model_retrained.evaluate(Xtest, ytest, verbose=0)[1]
            reward = new_acc - base_acc
            rewards.append(reward)

            print('reward: ' + str(reward))
        #retrain the oracle with the choosen sample and the real reward

        #TODO:lets do it in offline fashion

        oracle.fit(ypred_unseen[winner_idx_list], rewards)

    return('done')








