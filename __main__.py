import sys
import numpy as np
import tensorflow as tf

from .TransferLearning import fetch_data, fine_tune, retrain, savemodel, loadmodel
from .Testing import tester, experiment
from .ActiveLearning import seperation
from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn, mutural_info_uniform_fn, diff_uniform_fn
from .MetricsMethod import metric_method, mutural_info_method

epochs = 50
epochs_retrain = 5
batch_size = 128
retrain_batch = 150

def __main__():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.config.experimental.list_physical_devices('GPU')
    tf.device('/device:GPU:0')

    #Xtrain, ytrain = fetch_data('train')
    #model = fine_tune(Xtrain,ytrain,epochs,batch_size)[0]
    #del Xtrain, ytrain

    model = loadmodel('model_50epochs')

    print(experiment(model,epochs_retrain,retrain_batch,batch_size,[mutural_info_method,random_fn,diff_uniform_fn,mutural_info_uniform_fn,metric_method]))

if __name__ == "__main__":
    __main__()


#todo: balance out classes during training process
#a = np.argmax(ytrain, axis=1)
#unique, counts = np.unique(a, return_counts=True)
#print(dict(zip(unique, counts)))






