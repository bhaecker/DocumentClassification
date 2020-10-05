import sys
import numpy as np
import imageio
import cv2

def convert():
    for set in ['test','unseen','train']:
        print(set)
        file = open('./Data2/labels/'+set+'.txt',"r")
        lines = ['./Data2/images/'+line.rstrip('\n') for line in file]
        file_paths= [line.split()[0] for line in lines]
        labels = [line.split()[1] for line in lines]

        image = cv2.resize(imageio.imread(file_paths[0]), (244, 244))
        image = np.asarray(image)
        image = np.stack((image[:][:][:], image[:][:][:], image[:][:][:]))#, axis=3)
        X = image.astype('float16')

        for path in file_paths[1:]:

            try:
                image = cv2.resize(imageio.imread(path), (244, 244))
                image = np.asarray(image)
                image = np.stack((image[:][:][:], image[:][:][:], image[:][:][:]))#, axis=3)
                image = image.astype('float16')
                X = np.concatenate((X, image))



            except:
                print(path)

        np.save(set+'.npy', X)
        labels = np.asarray(labels)
        np.save(set+'_labels.npy',labels)
        del X

