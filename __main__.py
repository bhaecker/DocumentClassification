from .TransferLearning import fetch_data, fine_tune, retrain, savemodel, loadmodel
from .Testing import tester, experiment
from .ActiveLearning import seperation
from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn
from .MetricsMethod import metric_method


import tensorflow as tf
import sys
import numpy as np

epochs = 60
epochs_retrain = 5
batch_size = 128
retrain_batch = 200

def __main__():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.config.experimental.list_physical_devices('GPU')
    tf.device('/device:GPU:0')

    Xtrain, ytrain = fetch_data('train')
    #model = fine_tune(Xtrain,ytrain,epochs,batch_size)[0]

    #savemodel(model,'testmodel')
    model = loadmodel('model_60epochs')

    print(experiment(model,epochs_retrain,retrain_batch,batch_size,[metric_method,entropy_fn,least_confident_fn, margin_sampling_fn, random_fn]))


    sys.exit()


    Xtrain,ytrain = fetch_data('train')

    model_base,history_topDense,history_all = fine_tune(Xtrain,ytrain,epochs,batch_size)

    np.save('history_topDense.npy',history_topDense.history)
    np.save('history_all.npy', history_all.history)
    print('history saved')

    #test the model
    print('***base line performance***')
    Xtest,ytest, = fetch_data('test')
    ypred = tester(Xtest,ytest,model_base)

    #get unseen data
    Xunseen,yunseen = fetch_data('unseen')

    print('***random choosen samples***')
    Xwinner, ywinner, Xloser, yloser = seperation(Xunseen,yunseen,model_base,retrain_batch,random_fn)
    model_random, history = retrain(model_base,epochs,32,Xwinner,ywinner)
    tester(Xtest,ytest,model_random)

    print('***highest least confident samples***')
    Xwinner, ywinner, Xloser, yloser = seperation(Xunseen,yunseen,model_base,retrain_batch,least_confident_fn)
    model_ent, history = retrain(model_base,epochs,32,Xwinner,ywinner)
    tester(Xtest,ytest,model_ent)

    print('***highest entropy samples***')
    Xwinner, ywinner, Xloser, yloser = seperation(Xunseen,yunseen,model_base,retrain_batch,entropy_fn)
    model_ent, history = retrain(model_base,epochs,32,Xwinner,ywinner)
    tester(Xtest,ytest,model_ent)

if __name__ == "__main__":
    __main__()


#todo: balance out classes during training process
#a = np.argmax(ytrain, axis=1)
#unique, counts = np.unique(a, return_counts=True)
#print(dict(zip(unique, counts)))






