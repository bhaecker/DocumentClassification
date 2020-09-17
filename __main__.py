import sys
import numpy as np
import tensorflow as tf
import collections
from tensorflow.keras.models import load_model

from .TransferLearning import fetch_data, fine_tune, retrain#, savemodel, loadmodel
from .Testing import tester, experiment
from .ActiveLearning import seperation
from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn, mutural_info_uniform_fn, diff_uniform_fn
from .MetricsMethod import metric_method, diversity_method, diversity_images_method, diversity_images_balanced_method #mutural_info_method too slow
from .RandomForest import RandomForest_method, RandomForest_fn, RandomForestRegressor_pretraining
from .Qlearning import RL_model, train_RL_model, RL_CNN_method, RL_human_method
from .ContextualBandits import ContextualAdaptiveGreedy_method, ContextualAdaptiveGreedy_RLmodel_method, pretrain_oracle, pretrain_RLmodel_oracle


#set seeds for reproducability
np.random.seed(42)
tf.random.set_seed(42)
#epochs = 100
epochs_retrain = 10
batch_size = 128
retrain_batch = 100

#number_games = 200

#ededededededededede

def __main__():

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #TODO: TRAIN pretrain_RLmodel_oracle ON ALL TRAINING DATA
    #Xtrain, ytrain = fetch_data('train')
    CNN_model = load_model('model_100_epochs.h5')
    #Rel_model = load_model('RL_model.h5')
    pretrain_RLmodel_oracle(CNN_model,100)

    #train_RL_model(Xtrain, ytrain, Rel_model, CNN_model, 100)

    list_methods = [ContextualAdaptiveGreedy_RLmodel_method,margin_sampling_fn]
    experiment(CNN_model,epochs_retrain,retrain_batch,batch_size,list_methods)

if __name__ == "__main__":
    __main__()






