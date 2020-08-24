from sklearn.metrics import mutual_info_score,normalized_mutual_info_score
from numpy import linalg as LA
import numpy as np
from TransferLearning import fetch_data, loadmodel

#a = np.array([0,1,3])
#b = np.array([0,2,3])
#print(LA.norm(b-a))
#print(normalized_mutual_info_score([0.1,0.1,0.1],[0,1,0]))
#print(normalized_mutual_info_score([1,1,1],[0,0,0]))

#store predictions as they come in

def f(X,y,number_samples,model):
    '''

    '''
    if np.shape(X)[0] <= number_samples:
        return(X,y,X,y)
    if type(model) == str:
        model = loadmodel(model)
    Ypred = model.predict(X)
    #print(Ypred)
    matrix = np.zeros((np.shape(Ypred)[0],np.shape(Ypred)[0]))
    for row, ypred_a in enumerate(Ypred):
        #print(row)
        for column, ypred_b in enumerate(Ypred):
            matrix[row][column] = LA.norm(ypred_a-ypred_b)
    print('matrix')
    print(matrix)
    #todo: include samples with highest distance until number_samples is reached. Attention: Indicies needed for split
    return()#X,y,X,y)



X,y = fetch_data('unseen')
X,y = X[:20],y[:20]
model = 'model_40epochs'
print(f(X,y,10,model))