import sys
import gym
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, GlobalAveragePooling2D, InputLayer, Input, Concatenate, Conv2D, Flatten, Dense

from tensorflow.keras.utils import plot_model

from TransferLearning import fetch_data, loadmodel
#from ActiveLearning import loadmodel

# loss
cross_entropies = tf.losses.softmax_cross_entropy(
    onehot_labels=tf.one_hot(actions, 3), logits=Ylogits)

loss = tf.reduce_sum(rewards * cross_entropies)

# training operation
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.99)
train_op = optimizer.minimize(loss)

def RL_model(number_classes):

    #CNN for image processing
    image_input = Input((244, 244, 3)) #same size as in CNN Model or numpy array of images
    conv_layer = Conv2D(16, (7, 7))(image_input)
    pool_layer = MaxPooling2D(pool_size=(2, 2))(conv_layer)
    conv_layer = Conv2D(32, (5, 5))(pool_layer)
    pool_layer = MaxPooling2D(pool_size=(3, 3))(conv_layer)
    conv_layer = Conv2D(64, (3, 3))(pool_layer)
    pool_layer = MaxPooling2D(pool_size=(5, 5))(conv_layer)
    conv_layer = Conv2D(64, (3, 3))(pool_layer)
    flat_layer = Flatten()(conv_layer)

    #predictions from classification model
    prediction_input = Input((number_classes,))

    concat_layer = Concatenate()([prediction_input, flat_layer])
    dense_layer = Dense(256, activation="relu")(concat_layer)
    dropout_layer = Dropout(0.3)(dense_layer)
    dense_layer = Dense(256, activation="relu")(dropout_layer)
    dropout_layer = Dropout(0.3)(dense_layer)
    dense_layer = Dense(256, activation="relu")(dropout_layer)
    dropout_layer = Dropout(0.3)(dense_layer)
    output_layer = Dense(number_classes, activation="softmax")(dropout_layer)

    model = Model(inputs=[image_input, prediction_input], outputs=output_layer)

    return(model)




model = RL_model(10)

#plot_model(model,  to_file='model.png',show_shapes=True, show_layer_names=True)

print(model.summary())

model.compile(optimizer='Adam', loss='categorical_crossentropy',metrics=['accuracy'])

# train the model on train data for a few epochs
Xtrain,ytrain = fetch_data('train')
Xtrain,ytrain = Xtrain[:100],ytrain[:100]

ML_model = loadmodel('model_40epochs')

ypred = ML_model.predict(Xtrain)
print('predictions made')
batch_size = 128
epochs= 30

history = model.fit(x=[Xtrain,ypred],y=ytrain,
            validation_split=0.2,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



sys.exit()

def q_decider(own_model,ML_model,Xunssen,yunseen,number_sampels):
    if type(own_model) == str:
        own_model = loadmodel(own_model)
    if type(ML_model) == str:
        ML_model = loadmodel(ML_model)


    #request_label is hardcoded



    return(Xwinner,ywinner,Xloser,yloser)






def naive_sum_reward_agent(env, num_episodes=100):
    # this is the table that will hold our summated rewards for
    # each action in each state
    r_table = np.zeros((5, 2))
    for g in range(num_episodes):
        s = env.reset()
        done = False
        while not done:
            if np.sum(r_table[s, :]) == 0:
                # make a random selection of actions
                a = np.random.randint(0, 2)
            else:
                # select the action with highest cummulative reward
                a = np.argmax(r_table[s, :])
            #print(s,a)
            new_s, r, done, _ = env.step(a)
            r_table[s, a] += r
            s = new_s
    return r_table

print(naive_sum_reward_agent(env))

def eps_greedy_q_learning_with_table(env, num_episodes=100):
    q_table = np.zeros((5, 2))
    y = 0.95
    eps = 0.5
    lr = 0.8
    decay_factor = 0.999
    for i in range(num_episodes):
        s = env.reset()
        eps *= decay_factor
        done = False
        while not done:
            # select the action with highest cummulative reward
            if np.random.random() < eps or np.sum(q_table[s, :]) == 0:
                a = np.random.randint(0, 2)
            else:
                a = np.argmax(q_table[s, :])
            # pdb.set_trace()
            new_s, r, done, _ = env.step(a)
            q_table[s, a] += r + lr * (y * np.max(q_table[new_s, :]) - q_table[s, a])
            s = new_s
    return q_table

print(eps_greedy_q_learning_with_table(env))

model = Sequential()
model.add(InputLayer(batch_input_shape=(1, 5)))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(2, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['mae'])


num_episodes = 2
y = 0.95
eps = 0.5
decay_factor = 0.999
r_avg_list = []
for i in range(num_episodes):
    s = env.reset()
    eps *= decay_factor
    if i % 100 == 0:
        print("Episode {} of {}".format(i + 1, num_episodes))
    done = False
    r_sum = 0
    for j in range(2):
        if np.random.random() < eps:
            a = np.random.randint(0, 2)
        else:
            a = np.argmax(model.predict(np.identity(5)[s:s + 1]))
            print(model.predict(np.identity(5)[s:s + 1]))
        new_s, r, done, _ = env.step(a)
        print(model.predict(np.identity(5)[new_s:new_s + 1]))
        target = r + y * np.max(model.predict(np.identity(5)[new_s:new_s + 1]))
        print(target)
        target_vec = model.predict(np.identity(5)[s:s + 1])[0]
        print(target_vec)
        target_vec[a] = target
        print(target_vec)
        print(np.identity(5)[s:s + 1])
        print(target_vec.reshape(-1, 2))
        model.fit(np.identity(5)[s:s + 1], target_vec.reshape(-1, 2), epochs=1, verbose=0)
        s = new_s
        r_sum += r
    r_avg_list.append(r_sum / 1000)
print(r_avg_list)