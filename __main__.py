import sys
import numpy as np
import tensorflow as tf
import collections
from tensorflow.keras.models import load_model

from .TransferLearning import fetch_data, fine_tune, retrain#, savemodel, loadmodel
from .Testing import tester, experiment_accumulated, experiment_single
from .ActiveLearning import seperation
from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn, mutural_info_uniform_fn, diff_uniform_fn
from .MetricsMethod import metric_method, diversity_method, diversity_images_balanced_method #mutural_info_method and diversity_images_method too slow
from .RandomForest import RandomForest_method, RandomForest_fn, RandomForestRegressor_pretraining
from .Qlearning import epsgreedy_dual_oracle, epsgreedy_mono_oracle, RL_CNN_method, RL_human_method
from .ContextualBandits import ContextualAdaptiveGreedy_mono_algo, ContextualAdaptiveGreedy_dual_algo
from .Backbone import RL_model_dual, RL_model_mono, pretrain_dual_oracle, pretrain_mono_oracle
from .ContextualDiversity import bob_contextual_diversity_method, random_contextual_diversity_method, random_contextual_diversity_method_numberoption, experiment_CD, genetic_contextual_diversity_method

tf.random.set_seed(42)
np.random.seed(42)

#epochs = 100
epochs_retrain = 10
batch_size = 128
retrain_batch = 100
#number_games = 200

def __main__():
    #TODO:Early Stopping!
    #print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #X = None
    #model = None
    #number_samples = 3333
    #number_features = 10

    # random numbers as data set
    #y1 = np.random.random((number_samples, number_features))
    #y2 = np.random.beta(a=0.1,b=0.9,size=(number_samples, number_features))
    #y3 = np.random.beta(a=0.01, b=0.1, size=(number_samples, number_features))
    #y = np.concatenate((y1,y2),axis=0)
    #y = np.concatenate((y, y3),axis=0)
    #np.random.shuffle(y)

    #random_contextual_diversity_method(X, y, retrain_batch, model)
    #bob_contextual_diversity_method(X, y, retrain_batch, model)
    setsize_list = [20,30,40]
    experiment_CD('model_101_epochs.h5',epochs_retrain,retrain_batch,batch_size,setsize_list)
    #list_methods = [entropy_fn,RL_human_method]
    #experiment_accumulated('model_101_epochs.h5',epochs_retrain,retrain_batch,batch_size,list_methods)

if __name__ == "__main__":
    __main__()