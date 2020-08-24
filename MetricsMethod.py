from sklearn.metrics import mutual_info_score,normalized_mutual_info_score
from numpy import linalg as LA
import numpy as np
from .TransferLearning import fetch_data, loadmodel


def metric_method(X,y,number_samples,model):
    '''

    '''
    if np.shape(X)[0] <= number_samples:
        return(X,y,X,y)
    if type(model) == str:
        model = loadmodel(model)
    Ypred = model.predict(X)

    #matrix = np.zeros((np.shape(Ypred)[0],np.shape(Ypred)[0]))
    distance_list = []
    for row, ypred_a in enumerate(Ypred):
        #print(row)
        for column, ypred_b in enumerate(Ypred):
            #matrix[row][column] = LA.norm(ypred_a-ypred_b)
            #fill index_list
            if row != column:
                current_distance = LA.norm(ypred_a - ypred_b)
                distance_list.append((current_distance,(row,column)))

    distance_list = sorted(distance_list, key =lambda x: x[0], reverse=True)
    #get the indices coresponding to the distances in the right order
    index_list = [index[1][i] for index in distance_list for i in range(0,2)]
    #shorten the list to desired length without duplicates
    n_farest = []
    i = 0
    while len(n_farest)+1 <= number_samples:
        if index_list[i] not in n_farest:
            n_farest = n_farest + [index_list[i]]
        i += 1
    #seperate unseen data in winner and looser data set by the indices
    Xwinner = X[n_farest, :, :]
    ywinner = y[n_farest]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[n_farest] = False
    Xloser = X[mask, :, :]
    yloser = y[mask]

    return(Xwinner, ywinner, Xloser, yloser)

