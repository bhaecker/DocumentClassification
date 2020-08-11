import sys
import math
import glob
import os
import numpy as np
import os.path as path
import imageio
import cv2


#load data for fine tuning

FILE_DIRECTORY = 'DocumentClassification/Data'
labels = ['ADVE', 'Email', 'Form', 'Letter', 'Memo', 'News', 'Note', 'Report', 'Resume', 'Scientific']


def make_split(split):

    for counter, label in enumerate(labels):
        #fetch all images of a label
        file_paths = glob.glob(path.join(FILE_DIRECTORY+'/Tobacco/'+label, '*.jpg'))

        #split the collection of images with respect to input
        length = len(file_paths)

        train_array_split = file_paths[:math.floor(length*split[0]/100)]
        test_array_split = file_paths[math.floor(length*split[0]/100):math.floor(length*(split[0]+split[1]) / 100)]
        unseen_array_split = file_paths[math.floor(length * (split[0]+split[1]) / 100):]

        #save images as arrays in the corresponding directories
        #make sure the array has the right size for the network
        images = [cv2.resize(imageio.imread(path),(244, 244)) for path in train_array_split]
        images = np.asarray(images)
        np.save(FILE_DIRECTORY + '/Tobacco_train/' + label + '.npy', images)

        images = [cv2.resize(imageio.imread(path), (244, 244)) for path in test_array_split]
        images = np.asarray(images)
        np.save(FILE_DIRECTORY + '/Tobacco_test/' + label + '.npy', images)

        images = [cv2.resize(imageio.imread(path), (244, 244)) for path in unseen_array_split]
        images = np.asarray(images)
        np.save(FILE_DIRECTORY + '/Tobacco_unseen/' + label + '.npy', images)

        #label the classes
        target = np.zeros(len(labels))
        target[counter] = 1
        np.save(FILE_DIRECTORY + '/Tobacco_train/' + label + '_target.npy', target)
        np.save(FILE_DIRECTORY + '/Tobacco_test/' + label + '_target.npy', target)
        np.save(FILE_DIRECTORY + '/Tobacco_unseen/' + label + '_target.npy', target)


split = [60,20,20]

