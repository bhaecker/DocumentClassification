import sys
import numpy as np
import tensorflow as tf
import collections

from .TransferLearning import fetch_data, fine_tune, retrain, savemodel, loadmodel
from .Testing import tester, experiment
from .ActiveLearning import seperation
from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn, mutural_info_uniform_fn, diff_uniform_fn
from .MetricsMethod import metric_method, mutural_info_method, diversity_method
from .RandomForest import RandomForest_method, RandomForest_fn, RandomForestRegressor_pretraining

epochs = 100
epochs_retrain = 5
batch_size = 128
retrain_batch = 100

def __main__():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.config.experimental.list_physical_devices('GPU')
    tf.device('/device:GPU:0')

    Xtrain, ytrain = fetch_data('train')
    Xtrain, ytrain = Xtrain[:300], ytrain[:300]
    class_distribution = collections.Counter(np.where(ytrain == 1)[1])
    print(class_distribution)

    sys.exit()
    #RandomForest_fn()

    #model = fine_tune(Xtrain,ytrain,epochs,batch_size)[0]
    #del Xtrain, ytrain
    #Xunseen, _ = fetch_data('unseen')
    model = loadmodel('model_100epochs')
    #yunseen_pred = model.predict(Xunseen)
    #for smallunseen in yunseen_pred:
    #   print(RandomForest_fn(smallunseen))
    #todo: scores are too similar, it always chooes first class then

    #RandomForestRegressor_pretraining(Xtrain, ytrain,model,25)


    method_list = [RandomForest_fn,margin_sampling_fn,diversity_method,RandomForest_method,metric_method]
    print(experiment(model,epochs_retrain,retrain_batch,batch_size,method_list))

if __name__ == "__main__":
    __main__()


#todo: balance out classes during training process
#a = np.argmax(ytrain, axis=1)
#unique, counts = np.unique(a, return_counts=True)
#print(dict(zip(unique, counts)))






