#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:57:18 2019

@author: jding
"""

#import tensorflow as tf
import numpy as np
#import UNet.architecture as models
#import UNet_code.UNet.queue_input as data_read_test
import sys
import os
#from os.path import isfile, join
#import scipy.misc
#import UNet_code.UNet.UNet_functions as uf
#import matplotlib.pylab as plt
import scipy.io as sio 
import random
from random import randint
import cv2
from tensorflow.keras import utils


class DataGenerator(utils.Sequence):
    def __init__(self, dataset, batch_size, NUM_CLASSES,shuffle=True):
        
        self.data=dataset.image_all
        self.batch_size = batch_size
        self.nClasses = NUM_CLASSES
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def __len__(self):
        return int(self.data.shape[0] / self.batch_size)

    def on_epoch_end(self):
        self.indexes = np.arange(self.data.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        all_slices=np.arange(self.data.shape[0])
        slices_temp = [all_slices[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(slices_temp)

        return X, y

    def __data_generation(self, slices_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.data.shape[1], self.data.shape[2], 2))
        y = np.empty((self.batch_size, self.data.shape[1], self.data.shape[2], 1))
       
        # Generate data
       
        X = self.data[slices_temp,:,:,0:2]

        # Store class
        y = self.data[slices_temp,:,:,2]
        y[y>7] = 0

        #y=np.expand_dims(y, axis=-1)

        X=X.astype('float32')
        y=y.astype('int32')
            
            #normalization 
        for i in range(X.shape[0]):
            #X[i] = (X[i]-np.min(X[i]))/(np.absolute(np.max(X[i])-np.min(X[i])))
            X[i,:,:,0] = (X[i,:,:,0]-np.min(X[i,:,:,0]))/(np.absolute(np.max(X[i,:,:,0])-np.min(X[i,:,:,0])))
            #X[i,:,:,1] = (X[i,:,:,1]-np.min(X[i,:,:,1]))/(np.absolute(np.max(X[i,:,:,1])-np.min(X[i,:,:,1])))
        #print('\nnormalized')
   
        return X, utils.to_categorical(y, num_classes=self.nClasses)
