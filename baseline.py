from scipy.stats import entropy
from sklearn.metrics import normalized_mutual_info_score,v_measure_score
import numpy as np
import random

def entropy_fn(annotation_vector):
    if sum(annotation_vector) == 0:
        return 0
    return entropy(annotation_vector, base = len(annotation_vector))

#def kullback_leibler_divergence_fn(annotation_vector):

# 1 - max
# annotation_vector with highest score is annotation_vector
# that has smallest maximum value
def least_confident_fn(annotation_vector):
    return 1 - max(annotation_vector)

# 1 - (max1 - max2) 
# annotation_vector with highest score is annotation_vector
# that has smallest difference between first and second maximum value
def margin_sampling_fn(annotation_vector):
    sorted_annotation_vector = sorted(annotation_vector, reverse = True)
    if (len(sorted_annotation_vector) >= 2 ):
        return 1 - (sorted_annotation_vector[0] - sorted_annotation_vector[1])
    else:
        return 0

def random_fn(annotation_vector):
    return random.uniform(0,1)

def mutural_info_normal_fn(annotation_vector):
    vector_sum = sum(annotation_vector)
    normal_vector = np.full_like(annotation_vector,vector_sum/len(annotation_vector))
    return(1-normalized_mutual_info_score(annotation_vector,normal_vector))#,average_method='arithmetic'))

def diff_normal_fn(annotation_vector):

    vector_sum = sum(np.array(annotation_vector))
    norm_ann_vec = annotation_vector/vector_sum
    #print(norm_ann_vec)
    length = len(annotation_vector)
    normal_vector = np.full_like(annotation_vector,1/length)
    #print(normal_vector)
    diff_vector = abs(norm_ann_vec-normal_vector)

    return(-sum(diff_vector))

#print(diff_normal_fn([4,0.1,0.2]))


#print(normalized_mutual_info_score([0,100,0],[0,100,0]))

