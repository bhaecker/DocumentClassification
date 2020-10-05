import sys
import numpy as np
from scipy.stats import entropy
from scipy.special import binom
from tensorflow.keras.models import load_model


def w(P_r,epsilon=0.001):
    '''
    w_r Shannon entropy in paper

    '''

    return(entropy(P_r) + epsilon)

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

    if np.argmax(yunseen_pred1) == np.argmax(yunseen_pred2):
        P_c_1 = P_c(np.array([yunseen_pred1]), np.argmax(yunseen_pred1))
        P_c_2 = P_c(np.array([yunseen_pred2]), np.argmax(yunseen_pred1))
        #print(P_c_1)
        #print(P_c_2)
        pairwise_diversity = 0.5*entropy(P_c_1,P_c_2)+0.5*entropy(P_c_2,P_c_1)
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
    number_trials = 20
    if type(model) == str:
        model = load_model(model)
    ypred = model.predict(X)
    pool_size = np.shape(ypred)[0]
    idx_old = np.random.randint(pool_size, size=number_samples)
    ypred_old = ypred[idx_old]
    diversity_old = diversity(ypred_old)
    print(diversity_old)
    for trial in number_trials:
        print('trail: '+str(trial))
        idx_new = np.random.randint(pool_size, size=number_samples)
        ypred_new = ypred[idx_new]
        diversity_new = diversity(ypred_new)
        if diversity_new > diversity_old:
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


