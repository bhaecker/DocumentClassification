import sys
import numpy as np
from random import sample
from scipy.stats import entropy
from scipy.special import binom
from tensorflow.keras.models import load_model


##test out Contextual Diversity:
import pandas as pd
from .TransferLearning import fetch_data, fine_tune, retrain, concate#, savemodel, loadmodel
from .Testing import tester

#todo: just take the short formula my guy
#todo: experiment with low and high CD and see if performance improves
#todo: if its just chose the subset not randomly but by looking at the vector

def w(P_r):
    '''
    w_r Shannon entropy in paper
    not needed since (1) in paper is not needed

    '''

    #P_r = P_rr + epsilon
    epsilon = 0.001
    P_r[P_r <= 0] = epsilon

    shannon_entropy = -np.sum(P_r*np.log2(P_r))
    return(shannon_entropy)

def P_c(yunseen_pred,label):
    '''
    P_i^c in paper

    for a single prediction vector this will just output the prediction vector itself
    '''

    position = np.where(np.argmax(yunseen_pred,axis=1) == label,True,False)

    predictions_with_predicted_label = yunseen_pred[position]
    number_predictions_with_predicted_label = predictions_with_predicted_label.shape[0]
    if number_predictions_with_predicted_label == 0:
        return('label was not predicted')
    counter = np.zeros(10)
    denominator = 0

    for prediction_vector in predictions_with_predicted_label:
        counter += w(prediction_vector)*prediction_vector
        denominator += w(prediction_vector)
        #print(w(prediction_vector))
        #print(counter)
        #print(denominator)
    return(1/(number_predictions_with_predicted_label)*counter/denominator)


def diversity_pairwise(yunseen_pred1,yunseen_pred2):
    '''
    d_[I_1,I_2] in paper

    '''
    epsilon = 0.001
    if np.argmax(yunseen_pred1) == np.argmax(yunseen_pred2):
        P_c_1 = P_c(np.array([yunseen_pred1]), np.argmax(yunseen_pred1))
        P_c_2 = P_c(np.array([yunseen_pred2]), np.argmax(yunseen_pred1))
        P_c_1[P_c_1 <= 0] = epsilon
        P_c_2[P_c_2 <= 0] = epsilon
        pairwise_diversity = 0.5*entropy(P_c_1,P_c_2)+0.5*entropy(P_c_2,P_c_1)
        #if pairwise_diversity == float("inf"):
            #print(P_c_1,P_c_2)
        return(pairwise_diversity)
    else:
        return(0)

def diversity(yunseen_pred):
    aggregate_diversity = 0
    for yunssen_vec1 in yunseen_pred:
        for yunssen_vec2 in yunseen_pred:
            aggregate_diversity += diversity_pairwise(yunssen_vec1,yunssen_vec2)

    return(aggregate_diversity)


def random_contextual_diversity_method(X,y,number_samples,model):
    if np.shape(X)[0] <= number_samples:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return(X,y,X_empty,y_empty)

    number_trials = 20
    if type(model) == str:
        model = load_model(model)
    ypred = model.predict(X)
    ypred = y
    pool_size = np.shape(ypred)[0]
    idx_old = np.random.randint(pool_size, size=number_samples)
    ypred_old = ypred[idx_old]
    diversity_old = diversity(ypred_old)
    print(diversity_old)
    for trial in range(number_trials):
        print('trail: '+str(trial))
        idx_new = np.random.randint(pool_size, size=number_samples)
        ypred_new = ypred[idx_new]
        diversity_new = diversity(ypred_new)
        if diversity_new > diversity_old and diversity_new != float("inf"):
            diversity_old = diversity_new
            idx_old = idx_new
    print(diversity_old)

    # seperate unseen data in winner and looser data set by the indices
    Xwinner = X[idx_old, :, :]
    ywinner = y[idx_old]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[idx_old] = False
    Xloser = X[mask, :, :]
    yloser = y[mask]

    return(Xwinner, ywinner, Xloser, yloser)


def genetic_contextual_diversity_method(X,y,number_samples,model):
    #if np.shape(X)[0] <= number_samples:
     #   X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
      #  y_empty = np.empty([0, np.shape(y)[1]])
       # return(X,y,X_empty,y_empty)

    #if type(model) == str:
     #   model = load_model(model)
    #ypred = model.predict(X)

    ypred = y
    pool_size = np.shape(ypred)[0]

    pop_size = 10
    number_parents = 2
    number_rounds = 20

    idx_table = np.random.randint(pool_size, size=(pop_size,number_samples))
    #print(idx_table)
    fitness_list = [diversity(ypred[idx_table[idx]]) for idx in range(pop_size)]
    fitness_probability = fitness_list/sum(fitness_list)
    print(fitness_list)
    parents_table_idx = np.random.choice(range(pop_size),number_parents,replace=False,p=fitness_probability)
    mated_idx = set(list(idx_table[parents_table_idx[0]]) +list(idx_table[parents_table_idx[1]]))

    #mated_idx = np.unique(np.concatenate(idx_table[parents_table_idx[0]],idx_table[parents_table_idx[1]]))
    offspring_idx =  sample(mated_idx,number_samples)
    max_fitness_old = np.max(fitness_list)
    winner_idx_old = idx_table[np.argmax(fitness_list)]

    print(max_fitness_old)

    for round in range(number_rounds):
        print(round)
        idx_table = np.random.randint(pool_size, size=(pop_size, number_samples))
        idx_table[0] = offspring_idx
        #print(idx_table)
        fitness_list = [diversity(ypred[idx_table[idx]]) for idx in range(pop_size)]
        print(fitness_list)
        if np.argmax(fitness_list) == 0:
            print('offspring is the fittest')

        fitness_probability = fitness_list / sum(fitness_list)
        parents_table_idx = np.random.choice(range(pop_size), number_parents, replace=False, p=fitness_probability)
        mated_idx = set(list(idx_table[parents_table_idx[0]]) + list(idx_table[parents_table_idx[1]]))
        offspring_idx = sample(mated_idx, number_samples)
        max_fitness_new = np.max(fitness_list)
        winner_idx_new = idx_table[np.argmax(fitness_list)]
        #if abs(max_fitness_old - max_fitness_new) < 10:
         #   break
        if max_fitness_new > max_fitness_old:
            print('found new one')

            max_fitness_old = max_fitness_new
            winner_idx_old = winner_idx_new
        print(max_fitness_old)

    #winner_idx_old = np.array(win)
    # seperate unseen data in winner and looser data set by the indices
    Xwinner = X[winner_idx_old, :, :]
    ywinner = y[winner_idx_old]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[winner_idx_old] = False
    Xloser = X[mask, :, :]
    yloser = y[mask]

    return(Xwinner, ywinner, Xloser, yloser)


def bob_contextual_diversity_method(X,y,number_samples,model):
    #if np.shape(X)[0] <= number_samples:
     #   X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
      #  y_empty = np.empty([0, np.shape(y)[1]])
       # return(X,y,X_empty,y_empty)

    #if type(model) == str:
     #   model = load_model(model)
    #ypred = model.predict(X)

    #diversity_pairwise()

    ypred = y
    pool_size = np.shape(ypred)[0]
    indx_list = list(range(pool_size))
    indx_list_winner = list(sample(indx_list, 1))
    done = False
    while len(indx_list_winner) < number_samples:
        indx_list_winner_new = [indx_list_winner[-1]]
        for step in range(18):
            print('step: ',step)
            #preloop
            diversity_old = 0
            for indx_winner in indx_list_winner_new:
                winner_idx = indx_list[0]
                diversity_old += diversity_pairwise(y[winner_idx], y[indx_winner])
            for indx in indx_list[1:]:
                diversity_new = 0
                for indx_winner in indx_list_winner_new:
                    diversity_new += diversity_pairwise(y[indx],y[indx_winner])
                if diversity_new > diversity_old:
                    winner_idx = indx
                    diversity_old = diversity_new
            indx_list_winner_new = indx_list_winner_new + [winner_idx]
            indx_list.remove(winner_idx)
            if len(indx_list_winner_new) + len(indx_list_winner) == number_samples:
                indx_list_winner = indx_list_winner + indx_list_winner_new
                done = True
                break
        if done == True:
            break
        indx_list_winner = indx_list_winner + indx_list_winner_new
    print('finaly len: ',len(indx_list_winner))
    print(diversity(y[indx_list_winner]))
    # seperate unseen data in winner and looser data set by the indices

    #Xwinner = X[winner_idx, :, :]
    #ywinner = y[winner_idx]

    #mask = np.ones(X.shape[0], dtype=bool)
    #mask[winner_idx] = False
    #Xloser = X[mask, :, :]
    #yloser = y[mask]

    #return(Xwinner, ywinner, Xloser, yloser)



##test out Contextual Diversity:
###############


def random_contextual_diversity_method_numberoption(X,y,number_samples,model, number_trials):
    if np.shape(X)[0] <= number_samples:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return(X,y,X_empty,y_empty)

    if type(model) == str:
        print('start loading model')
        model = load_model('./DocumentClassification/'+model)
        print('finished loading model')
    ypred = model.predict(X)
    pool_size = np.shape(ypred)[0]
    idx_old = np.random.randint(pool_size, size=number_samples)
    ypred_old = ypred[idx_old]
    diversity_old = diversity(ypred_old)
    print(diversity_old)
    for trial in range(number_trials):
        print('trail: '+str(trial))
        idx_new = np.random.randint(pool_size, size=number_samples)
        ypred_new = ypred[idx_new]
        diversity_new = diversity(ypred_new)
        if diversity_new > diversity_old and diversity_new != float("inf"):
            diversity_old = diversity_new
            idx_old = idx_new
    print(diversity_old)

    # seperate unseen data in winner and looser data set by the indices
    Xwinner = X[idx_old, :, :]
    ywinner = y[idx_old]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[idx_old] = False
    Xloser = X[mask, :, :]
    yloser = y[mask]

    return(Xwinner, ywinner, Xloser, yloser, diversity_old)

def bob_contextual_diversity_method_setoption(X,y,number_samples,model,setsize):
    if np.shape(X)[0] <= number_samples:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return(X,y,X_empty,y_empty)

    if type(model) == str:
        model = load_model(model)

    ypred = model.predict(X)

    pool_size = np.shape(ypred)[0]
    indx_list = list(range(pool_size))
    indx_list_winner = list(sample(indx_list, 1))
    done = False
    while len(indx_list_winner) < number_samples:
        indx_list_winner_new = [indx_list_winner[-1]]
        for step in range(setsize):
            print('step: ',step)
            #preloop
            diversity_old = 0
            for indx_winner in indx_list_winner_new:
                winner_idx = indx_list[0]
                diversity_old += diversity_pairwise(y[winner_idx], y[indx_winner])
            for indx in indx_list[1:]:
                diversity_new = 0
                for indx_winner in indx_list_winner_new:
                    diversity_new += diversity_pairwise(y[indx],y[indx_winner])
                if diversity_new > diversity_old:
                    winner_idx = indx
                    diversity_old = diversity_new
            indx_list_winner_new = indx_list_winner_new + [winner_idx]
            indx_list.remove(winner_idx)
            if len(indx_list_winner_new) + len(indx_list_winner) == number_samples:
                indx_list_winner = indx_list_winner + indx_list_winner_new
                done = True
                break
        if done == True:
            break
        indx_list_winner = indx_list_winner + indx_list_winner_new
    print('final len: ',len(indx_list_winner))
    diversity_fin = diversity(ypred[indx_list_winner])
    # seperate unseen data in winner and looser data set by the indices
    print(diversity_fin)
    Xwinner = X[winner_idx, :, :]
    ywinner = y[winner_idx]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[winner_idx] = False
    Xloser = X[mask, :, :]
    yloser = y[mask]

    return(Xwinner, ywinner, Xloser, yloser, diversity_fin)


def experiment_CD(model_base_str, epochs_retrain, retrain_size, mini_batch_size, setsize_list):
    '''
    '''

    if type(model_base_str) != str:
        exit('please provide name of model as string')

    Xtest, ytest = fetch_data('test')
    Xtrain, ytrain = fetch_data('train')
    print('right data fetched')

    #base_performance = round(tester(Xtest, ytest, model_base_str)[0], 2)
    df = pd.DataFrame(columns=['CD score','accuracy'])

    Xunseen_orig, yunseen_orig = fetch_data('unseen')

    # shuffle the unseen data
    rng_state = np.random.get_state()
    np.random.shuffle(Xunseen_orig)
    np.random.set_state(rng_state)
    np.random.shuffle(yunseen_orig)

    # Xunseen_orig, yunseen_orig = Xunseen_orig[:200], yunseen_orig[:200]

    for idx, number_random_trial in enumerate(setsize_list):
        Xwinner, ywinner, _, _, diversity = bob_contextual_diversity_method_setoption(Xunseen_orig, yunseen_orig,retrain_size,model_base_str,setsize_list)

        # new trainings batch consists of old training samples plus the new unseen ones
        # Xtrain_new = np.concatenate((Xtrain, Xwinner), axis=0)
        # ytrain_new = np.concatenate((ytrain, ywinner),axis=0)

        Xtrain_new = concate(Xtrain, Xwinner)
        ytrain_new = concate(ytrain, ywinner)

        print('training data concatenated')

        model_base = load_model('./DocumentClassification/'+model_base_str)
        print('model loaded')
        model_new = retrain(model_base, epochs_retrain, mini_batch_size, Xtrain_new, ytrain_new)[0]
        accuracy = tester(Xtest, ytest, model_new)[0]

        df.at[idx, 'CD score'] =  diversity
        df.at[idx, 'accuracy'] = accuracy
        print(df)

        del model_new, model_base

    df.to_csv('RESULTS.csv', index=False)
    print(df)
    return (df)

