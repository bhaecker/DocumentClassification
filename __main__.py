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

tf.random.set_seed(42)
np.random.seed(42)

#epochs = 100
epochs_retrain = 2
batch_size = 128
retrain_batch = 200
#number_games = 200

def __main__():

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    #Xtrain, ytrain = fetch_data('train')
    #CNN_model = load_model('model_100_epochs.h5')

    #
    list_methods = [RL_CNN_method,RL_human_method,metric_method]
    experiment_single('model_100_epochs.h5',epochs_retrain,retrain_batch,batch_size,list_methods)

if __name__ == "__main__":
    __main__()






