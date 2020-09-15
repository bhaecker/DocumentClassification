import sys
import numpy as np
import tensorflow as tf
import collections

from .TransferLearning import fetch_data, fine_tune, retrain#, savemodel, loadmodel
from .Testing import tester, experiment
#from .ActiveLearning import seperation
#from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn, mutural_info_uniform_fn, diff_uniform_fn
#from .MetricsMethod import metric_method, mutural_info_method, diversity_method, diversity_images_method, diversity_images_balanced_method
#from .RandomForest import RandomForest_method, RandomForest_fn, RandomForestRegressor_pretraining
#from .Qlearning import RL_model, train_RL_model, RL_CNN_method, RL_human_method
#from .ContextualBandits import ContextualAdaptiveGreedy_method

epochs = 5
epochs_retrain = 10
batch_size = 128
#retrain_batch = 200

#number_games = 200

def __main__():

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    Xtrain, ytrain = fetch_data('train')
    CNN_model = fine_tune(Xtrain, ytrain,epochs,batch_size)[0]

    Xtest, ytest = fetch_data('test')
    tester(Xtest, ytest, CNN_model)

    print('resume training in session')
    Xunseen, yunseen = fetch_data('unseen')
    CNN_model_retrained = retrain(CNN_model,epochs_retrain,batch_size,Xunseen, yunseen)[0]
    tester(Xtest, ytest, CNN_model_retrained)
    del CNN_model,CNN_model_retrained

    print('resume training with reloaded model')
    CNN_model_retrained = retrain('model_5_epochs.h5', epochs_retrain, batch_size, Xunseen, yunseen)[0]
    tester(Xtest, ytest, CNN_model_retrained)

if __name__ == "__main__":
    __main__()






