import sys
import math
import glob
import os
import numpy as np
import os.path as path
import imageio
import cv2

#load data for fine tuning

FILE_DIRECTORY = 'Data/Tobacco/'
labels = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']


def make_split(split):

    for counter, label in enumerate(labels):
        #fetch all images of a label
        file_paths = glob.glob(os.path.join(FILE_DIRECTORY+label, '*.jpg'))
        #split the collection of images with respect to input
        length = len(file_paths)

        train_array_split = file_paths[:math.floor(length*split[0]/100)]
        test_array_split = file_paths[math.floor(length*split[0]/100):math.floor(length*(split[0]+split[1]) / 100)]
        unseen_array_split = file_paths[math.floor(length * (split[0]+split[1]) / 100):]

        #save images as arrays in the corresponding directories
        #make sure the array has the right size and type for the network
        images = [cv2.resize(imageio.imread(path),(244, 244)) for path in train_array_split]
        images = np.asarray(images)
        images = np.stack((images[:][:][:], images[:][:][:], images[:][:][:]), axis=3)
        images = images.astype('float16')
        np.save('Tobacco_train/' + label + '.npy', images)

        images = [cv2.resize(imageio.imread(path), (244, 244)) for path in test_array_split]
        images = np.asarray(images)
        images = np.stack((images[:][:][:], images[:][:][:], images[:][:][:]), axis=3)
        images = images.astype('float16')
        np.save('Tobacco_test/' + label + '.npy', images)

        images = [cv2.resize(imageio.imread(path), (244, 244)) for path in unseen_array_split]
        images = np.asarray(images)
        images = np.stack((images[:][:][:], images[:][:][:], images[:][:][:]), axis=3)
        images = images.astype('float16')
        np.save('Tobacco_unseen/' + label + '.npy', images)

        #label the classes
        target = np.zeros(len(labels))
        target[counter] = 1
        np.save('Tobacco_train/' + label + '_target.npy', target)
        np.save('Tobacco_test/' + label + '_target.npy', target)
        np.save('Tobacco_unseen/' + label + '_target.npy', target)

split = [60,20,20]


