# -----------------------------------------------------------
# Preprocessing Tobacco data set with names in data included
# -----------------------------------------------------------

import sys
import math
import glob
import os
import numpy as np
import os.path as path
import imageio
import cv2
import random

#load data for fine tuning

FILE_DIRECTORY = 'Data/Tobacco/'
labels = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']

np.random.seed(42)

def make_split(split):

    train_array_split_all = []
    test_array_split_all = []
    unseen_array_split_all = []

    for counter, label in enumerate(labels):
        #fetch all images of a label

        file_paths = glob.glob(os.path.join(FILE_DIRECTORY+label, '*.jpg'))

        #shuffle to aviod bias
        random.shuffle(file_paths)
        #split the collection of images with respect to input
        length = len(file_paths)

        train_array_split = file_paths[:math.floor(length*split[0]/100)]
        test_array_split = file_paths[math.floor(length*split[0]/100):math.floor(length*(split[0]+split[1]) / 100)]
        unseen_array_split = file_paths[math.floor(length * (split[0]+split[1]) / 100):]

        train_array_split_all.extend(train_array_split)
        test_array_split_all.extend(test_array_split)
        unseen_array_split_all.extend(unseen_array_split)


        #save images as arrays in the corresponding directories
        #make sure the array has the right size and type for the network
        images = [cv2.resize(imageio.imread(path),(244, 244)) for path in train_array_split]
        images = np.asarray(images)
        images = np.stack((images[:][:][:], images[:][:][:], images[:][:][:]), axis=3)
        images = images.astype('float16')
        np.save('Data/Tobacco_names_train/' + label + '.npy', images)

        images = [cv2.resize(imageio.imread(path), (244, 244)) for path in test_array_split]
        images = np.asarray(images)
        images = np.stack((images[:][:][:], images[:][:][:], images[:][:][:]), axis=3)
        images = images.astype('float16')
        np.save('Data/Tobacco_names_test/' + label + '.npy', images)

        images = [cv2.resize(imageio.imread(path), (244, 244)) for path in unseen_array_split]
        images = np.asarray(images)
        images = np.stack((images[:][:][:], images[:][:][:], images[:][:][:]), axis=3)
        images = images.astype('float16')
        np.save('Data/Tobacco_names_unseen/' + label + '.npy', images)

        #label the classes
        target = np.zeros(len(labels))
        target[counter] = 1
        np.save('Data/Tobacco_names_train/' + label + '_target.npy', target)
        np.save('Data/Tobacco_names_test/' + label + '_target.npy', target)
        np.save('Data/Tobacco_names_unseen/' + label + '_target.npy', target)

    with open("train_array_split_all.txt", "w") as f:
        for s in train_array_split_all:
            f.write(str(s) + "\n")
    with open("test_array_split_all.txt", "w") as f:
        for s in test_array_split_all:
            f.write(str(s) + "\n")
    with open("unseen_array_split_all.txt", "w") as f:
        for s in unseen_array_split_all:
            f.write(str(s) + "\n")

split = [60,20,20]


