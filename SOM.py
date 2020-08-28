import pandas as pd
import SimpSOM as sps
from sklearn.cluster import KMeans
import numpy as np

from TransferLearning import fetch_data, loadmodel

Xunseen,yunseen = fetch_data('unseen')
yunseen = yunseen[:10]
Xunseen = Xunseen[:10]
print(np.shape(Xunseen),np.shape(yunseen))
#train = np.concatenate((Xunseen,yunseen), axis = 0)
#print(np.shape(train))

model = loadmodel('model_40epochs')

ypred = model.predict(Xunseen)

#print(Xunseen)

net = sps.somNet(20, 20, ypred, PBC=True)
net.train(0.01, 500)

#net.nodes_graph(colnum=0,show=True)
net.diff_graph(show=True)
labels = [0,1,2,3,4,5,6,7,8,9]
#Project the datapoints on the new 2D network map.
net.project(yunseen,show= True ,labels=labels)

#Cluster the datapoints according to the Quality Threshold algorithm.
net.cluster(yunseen,show= True, type='qthresh')
