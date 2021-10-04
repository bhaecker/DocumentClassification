import sys
import numpy as np
import tensorflow as tf
import collections
from tensorflow.keras.models import load_model

from .TransferLearning import fetch_data, fine_tune, retrain#, savemodel, loadmodel
from .Testing import tester, experiment_accumulated, experiment_single,experiment_single_earlystopping,experiment_accumulated_earlystopping,experiment_accumulated_earlystopping_modelreuse
from .ActiveLearning import seperation
from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn, mutural_info_uniform_fn, diff_uniform_fn
from .MetricsMethod import vecnorm_metric_method, diversity_method, diversity_images_balanced_method, cosine_distance_method,cosine_distance_opti_method,vecnorm_metric_opti_method,wasserstein_distance_opti_method #mutural_info_method and diversity_images_method too slow
from .RandomForest import RandomForest_method, RandomForest_fn, RandomForestRegressor_pretraining
from .Qlearning import epsgreedy_dual_oracle, epsgreedy_mono_oracle, RL_CNN_method, RL_human_method, RL_dual_human_method, RL_dual_CNN_method
from .ContextualBandits import ContextualAdaptiveGreedy_mono_algo, ContextualAdaptiveGreedy_dual_algo
from .Backbone import RL_model_dual, RL_model_mono, pretrain_dual_oracle, pretrain_mono_oracle
from .ContextualDiversity import diversity_metric_opti_method,diversity_metric_method,bob_contextual_diversity_method_setoption_method,bob_contextual_diversity_method, random_contextual_diversity_method, random_contextual_diversity_method_numberoption, experiment_CD, genetic_contextual_diversity_method
from .XAI import get_all_layer_outputs, make_explantation_from_distances

tf.random.set_seed(42)
np.random.seed(42)

#epochs = 100
#epochs_retrain = 10
#batch_size = 128
#retrain_batch = 700
#number_games = 200

def __main__():
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
    #setsize_list = [20,30,40]
    #X,model = None,None

    #model = load_model("model_101_epochs.h5")
    #print('model loaded')
    #print(model.summary())

    #test_X,test_y = fetch_data('trans_test')
    train_X,train_y = fetch_data('trans_train')
    #unseen_X,unseen_y = fetch_data('trans_unseen')

    #print(np.shape(test_X))
    #print(np.shape(train_X))
    #print(np.shape(unseen_X))
    model = "model_20_epochs.h5"

    #mono_oracle = RL_model_mono(10,2)
    #dual_oracle = RL_model_dual(10,2)

    #epsgreedy_mono_oracle(train_X,train_y, mono_oracle, model, 20)
    #epsgreedy_dual_oracle(train_X, train_y, dual_oracle, model, 20)



    #pretrain_dual_oracle(dual_oracle,model,20)
    #pretrain_mono_oracle(mono_oracle,model,20)

    #print('done pretraining, start experiments')

    list_methods = [entropy_fn,wasserstein_distance_opti_method]
    #551 for equal AL batches
    experiment_accumulated_earlystopping(model,20,551,128,list_methods)

    sys.exit()


    accuracy_test,_= tester(test_X,test_y,model)
    print(accuracy_test)

    accuracy_unseen, _ = tester(unseen_X, unseen_y, model)
    print(accuracy_unseen)

    accuracy_train, _ = tester(train_X,train_y, model)
    print(accuracy_train)

    ex_samples_id_unseen = []
    with open("/newstorage4/bhaecker/unseen_array_split_all.txt", "r") as f:
        for line in f:
            ex_samples_id_unseen.append(str(line.strip()))

    idx_in_unseen = 200
    print(ex_samples_id_unseen[idx_in_unseen])
    sample = unseen_X[idx_in_unseen:idx_in_unseen+1,:]
    pred =model.predict(sample)
    print(np.argmax(pred))
    print(np.argmax(unseen_y[idx_in_unseen:idx_in_unseen+1][0]))
    make_explantation_from_distances(test_X,model,sample,10,"Conv2D")

if __name__ == "__main__":
    __main__()