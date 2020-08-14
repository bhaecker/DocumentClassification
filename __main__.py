from .TransferLearning import fetch_data, fine_tune, retrain
from .Testing import tester
from .ActiveLearning import seperation
from .baseline import entropy_fn, least_confident_fn, margin_sampling_fn, random_fn

import tensorflow as tf

epochs = 5

def __main__():
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    tf.config.experimental.list_physical_devices('GPU')
    tf.device('/device:GPU:0')

    Xtrain,ytrain = fetch_data('train')

    model_base = fine_tune(Xtrain,ytrain,epochs)

    #test the model
    print('***base line performance***')
    Xtest,ytest, = fetch_data('test')
    ypred = tester(Xtest,ytest,model_base)

    #get unseen data
    Xunseen,yunseen = fetch_data('unseen')

    print('***random choosen samples***')
    Xwinner, ywinner, Xloser, yloser = seperation(Xunseen,yunseen,model_base,25,random_fn)
    model_random = retrain(model_base,epochs,32,Xwinner,ywinner)
    tester(Xtest,ytest,model_random)

    print('***highest least confident samples***')
    Xwinner, ywinner, Xloser, yloser = seperation(Xunseen,yunseen,model_base,25,least_confident_fn)
    model_ent = retrain(model_base,epochs,32,Xwinner,ywinner)
    tester(Xtest,ytest,model_ent)

    print('***highest entropy samples***')
    Xwinner, ywinner, Xloser, yloser = seperation(Xunseen,yunseen,model_base,25,entropy_fn)
    model_ent = retrain(model_base,epochs,32,Xwinner,ywinner)
    tester(Xtest,ytest,model_ent)

if __name__ == "__main__":
    __main__()


#todo: balance out classes during training process
#a = np.argmax(ytrain, axis=1)
#unique, counts = np.unique(a, return_counts=True)
#print(dict(zip(unique, counts)))






