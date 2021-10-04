# -----------------------------------------------------------
# BONUS: Explainable AI with similar neuron activity during training -> see XAI repo for details
# -----------------------------------------------------------

import sys
import numpy as np
#from numpy import linalg as LA
from scipy.spatial.distance import cdist
from tensorflow.keras.models import Model


def get_all_layer_outputs(basic_model,training_data):
    '''
    returns a list of length number of hidden layers + 1 (for output layer),
    where every element is an array of the activation for each training sample for the respective layer
    '''
    intermediate_model = Model(inputs=basic_model.layers[0].input,
                              outputs=[l.output for l in basic_model.layers[1:]])

    activation_set = intermediate_model.predict(training_data)
    print(len(activation_set))
    #for activation in activation_set:
        #print(np.shape(activation))
    return activation_set

def make_explantation_from_distances(training_data,model,sample,number_explanations,importance):
    '''
    calculates the euclidean distance between every activation of sample and training data
    and returns number_explanations closest training samples
    linear_importance is float number between 0 and 1 and weights the layer output with liner weights starting from linear_importance to 1
    '''

    ex_samples_id_test = []
    with open("/newstorage4/bhaecker/test_array_split_all.txt", "r") as f:
        for line in f:
            ex_samples_id_test.append(str(line.strip()))




    layer_outputs_training = get_all_layer_outputs(model,training_data)
    layer_outputs_sample =  get_all_layer_outputs(model,sample)

    number_training_samples = np.shape(training_data)[0]
    number_layers = len(layer_outputs_sample)
    distance_array = np.empty(shape=(number_training_samples,number_layers))

    for idx, (layer_output_training, layer_output_sample) in enumerate(zip(layer_outputs_training,layer_outputs_sample)):

        shape_tuple_sample = list(np.shape(layer_output_sample))
        layer_output_sample_new = layer_output_sample.reshape(shape_tuple_sample[0],np.prod(shape_tuple_sample[1:]))
        shape_tuple_training = list(np.shape(layer_output_training))
        layer_output_training_new = layer_output_training.reshape(shape_tuple_training[0], np.prod(shape_tuple_training[1:]))
        distances = cdist(layer_output_training_new,layer_output_sample_new,metric='euclidean')
        distance_array[:,idx] = np.transpose(distances)

    if type(importance) == float or type(importance) == int:
        weights = np.linspace(start=importance, stop=1, num=number_layers)
    elif type(importance) == str:
        weights = [1 if layer.__class__.__name__ == importance else 0 for layer in model.layers]
        weights = np.asarray(weights[:-1])
    else:
        weights = np.ones(shape=number_layers)

    print(weights)
    print(distance_array)

    sum_weighted_distance_array = np.dot(distance_array,weights)

    #find 'number_explanations' smallest values !!!Attention, does not return them sorted
    idx_winner = np.argpartition(sum_weighted_distance_array, number_explanations)[:number_explanations]
    print(sum_weighted_distance_array)
    print(sum_weighted_distance_array[idx_winner])

    print(idx_winner)

    for idx in idx_winner:
        print(ex_samples_id_test[idx])
    explantation_samples = training_data[idx_winner]

    print("When the model 'saw'")
    np.save("sample.npy",sample)
    print('saved to sample.npy')
    print("it was 'thinking' the same as when it 'saw' following samples with importance to/" + str(importance))
    np.save("explantation_samples.npy", explantation_samples)
    print('saved to explantation_samples.npy')


    return 'done'





