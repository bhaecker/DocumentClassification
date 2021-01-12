import sys
import math
import random
import numpy as np
from numpy import linalg as LA
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mutual_info_score,normalized_mutual_info_score
from tensorflow.keras.models import load_model

from .TransferLearning import fetch_data

def vecnorm_metric_method(X,y,number_samples,model):
    '''
    gives back the samples, which predictions are most far away (with respect to L2 norm) in prediction space
    '''
    if np.shape(X)[0] <= number_samples:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return(X,y,X_empty,y_empty)
    if type(model) == str:
        model = load_model(model)
    Ypred = model.predict(X)

    #matrix = np.zeros((np.shape(Ypred)[0],np.shape(Ypred)[0]))
    distance_list = []
    for row, ypred_a in enumerate(Ypred):
        # TODO: Start from current row since matrix is semetric! check where is the mistake
        for column, ypred_b in enumerate(Ypred):
            #matrix[row][column] = LA.norm(ypred_a-ypred_b)
            #fill index_list
            if row != column:
                current_distance = LA.norm(ypred_a - ypred_b)
                distance_list.append((current_distance,(row,column)))

    distance_list = sorted(distance_list, key =lambda x: x[0], reverse=False)

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


def diversity_method(X,y,number_samples,model):
    '''
    #chosediversity
    '''
    if np.shape(X)[0] <= number_samples:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return (X, y, X_empty, y_empty)
    if type(model) == str:
        model = load_model(model)
    Ypred = model.predict(X)

    number_classes = np.shape(Ypred)[1]
    print(np.shape(Ypred))
    #make a list of lists for each class, containing the indicies of the samples, which have the highest prediction value for said class
    index_list_lists = [[] for _ in range(number_classes)]
    for row,ypred in enumerate(Ypred):
        index_list_lists[np.where(ypred == np.amax(ypred))[0][0]].append(row)
    #we want an equal amount of samples from each predicted class in our winner set
    samples_class = math.floor(number_samples/number_classes)
    #choose a random sample of samples if there are too many samples for one class
    winner_indices = []
    for index_list in index_list_lists:
        if len(index_list) > samples_class:
            winner_indices.extend(random.sample(index_list,samples_class))
        else:
            winner_indices.extend(index_list)
    #add randomly choosen samples if there are not enough for desired size
    if len(winner_indices) < number_samples:
        index_list_flatten =  [item for sublist in index_list_lists for item in sublist]
        indicies_left = [x for x in index_list_flatten if x not in winner_indices]
        winner_indices = winner_indices + random.sample(indicies_left,number_samples-len(winner_indices))

    # seperate unseen data in winner and looser data set by the indices
    Xwinner = X[winner_indices, :, :]
    ywinner = y[winner_indices]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[winner_indices] = False
    Xloser = X[mask, :, :]
    yloser = y[mask]

    return(Xwinner, ywinner, Xloser, yloser)


def diversity_images_balanced_method(X,y,number_samples,model):
    '''
    return samples, which are the most unsimilar (with respect to structural similarity) and belong to the same predicted class
    balance out the classes as well, as far as possible
    '''
    number_labels = 10
    if np.shape(X)[0] <= number_samples:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return (X, y, X_empty, y_empty)
    if type(model) == str:
        model = load_model(model)
    #make predictions
    Ypred = model.predict(X)

    #calculate the similarities between all images
    distance_list = []
    for label in range(number_labels):#number of labels
        for row,(X_row, y_pred_row)  in enumerate(zip(X,Ypred)):
            #since similarity is semteric, only compare from current image on
            for col, (X_col, y_pred_col)  in enumerate(zip(X[row+1:],Ypred[row+1:])):
                if np.argmax(y_pred_col) == label == np.argmax(y_pred_row):
                    current_distance = ssim(X_row[:,:,0],X_col[:,:,0])
                    distance_list.append((current_distance, (row, col+row+1, label)))
    #sort list from smallest values on (the higher the value, the more similar the images)
    distance_list = sorted(distance_list, key=lambda x: x[0], reverse=False)

    #get the number_samples farest images and balance out their classes
    n_farest = []
    labels_left = np.unique([item[1][2] for item in distance_list]).tolist()
    i = 0
    while len(n_farest) + 1 <= number_samples:
        #do not include duplicates and take samples from left over classes
        if distance_list[i][1][0] not in n_farest and distance_list[i][1][2] in labels_left:
            n_farest = n_farest + [distance_list[i][1][0]]
            labels_left.remove(distance_list[i][1][2])
        #reset the classes, when they are all used up or no sample is from the left over classes
        if labels_left == [] or i == len(distance_list)-1:
            labels_left = np.unique([item[1][2] for item in distance_list]).tolist()
        i = (i+1)%len(distance_list)

    #seperate unseen data in winner and looser data set by the indices
    Xwinner = X[n_farest, :, :]
    ywinner = y[n_farest]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[n_farest] = False
    Xloser = X[mask, :, :]
    yloser = y[mask]

    del distance_list,n_farest,mask,model

    return(Xwinner, ywinner, Xloser, yloser)

####################################################################################

def diversity_images_method(X,y,number_samples,model):#IMPORTANT TOO SLOW
    '''
    return samples, which are the most unsimilar (with respect to structural similarity)
    '''

    if np.shape(X)[0] <= number_samples:
        X_empty = np.empty([0, np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]])
        y_empty = np.empty([0, np.shape(y)[1]])
        return (X, y, X_empty, y_empty)
    #if type(model) == str:
     #   model = loadmodel(model)
    #make predictions
    #Ypred = model.predict(X)

    #calculate the similarities between all images
    distance_list = []

    for row,X_row  in enumerate(X):
        #since similarity is semteric, only compare from current image on
        for col, X_col in enumerate(X[row+1:]):
            current_distance = ssim(X_row[:,:,0],X_col[:,:,0])
            distance_list.append((current_distance, (row, col+row+1)))
    #sort list from smallest values on (the higher the value, the more similar the images)
    distance_list = sorted(distance_list, key=lambda x: x[0], reverse=False)
    #get the indices coresponding to the distances in the right order
    index_list = [index[1][i] for index in distance_list for i in range(0,2)]
    #shorten the list to desired length without duplicates
    n_farest = []
    i = 0
    while len(n_farest) + 1 <= number_samples:
        if index_list[i] not in n_farest:
            n_farest = n_farest + [index_list[i]]
        i += 1
    print(n_farest)
    #seperate unseen data in winner and looser data set by the indices
    Xwinner = X[n_farest, :, :]
    ywinner = y[n_farest]

    mask = np.ones(X.shape[0], dtype=bool)
    mask[n_farest] = False
    Xloser = X[mask, :, :]
    yloser = y[mask]

    del distance_list,n_farest,mask,index_list,model

    return(Xwinner, ywinner, Xloser, yloser)





def mutural_info_method(X,y,number_samples,model):
    #IMPORTANT: Too slow
    '''
    gives back the samples, which predictions differ the most (with respect to normalized_mutual_info_score)
    '''
    if np.shape(X)[0] <= number_samples:
        return(X,y,X,y)
    if type(model) == str:
        model = load_model(model)
    Ypred = model.predict(X)

    #matrix = np.zeros((np.shape(Ypred)[0],np.shape(Ypred)[0]))
    distance_list = []
    for row, ypred_a in enumerate(Ypred):
        #print(row)
        for column, ypred_b in enumerate(Ypred):
            #matrix[row][column] = LA.norm(ypred_a-ypred_b)
            #fill index_list
            if row != column:
                current_distance = normalized_mutual_info_score(ypred_a, ypred_b)
                distance_list.append((current_distance,(row,column)))

    distance_list = sorted(distance_list, key =lambda x: x[0], reverse=False)

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


