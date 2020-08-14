from scipy.stats import entropy
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