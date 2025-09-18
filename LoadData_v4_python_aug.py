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
#import cv2
#from tensorflow.keras import utils


class Dataloader(object):
    def __init__(self, mriDir):
        self.mriDir = mriDir

        self.all_images  = [x for x in sorted(os.listdir(self.mriDir)) if '.npy' in x]
        self.image_all = self.loadData()

    def loadData(self):

  #      selected_scans_idx = random.sample(range(0, len(all_images)-1), self.BatchSize3D)
        # randomly select BatchSize3D scans for later 2DBatch
        print('Loading scans')
        for i, mri_name in enumerate(self.all_images): #load scans in 3D volume

            if i % 200 == 0:
                print('Loading: '+str(round(float(i+1) / len(self.all_images) * 100,2)) +' %')
      #      mri_name=all_images[i]
            #print('\nLoading scans',self.BatchSize3D)
      #      print(mri_name)

            #mat_contents = sio.loadmat(self.mriDir + mri_name,squeeze_me=True)#.astype('float32')
            npy_contents = np.load(os.path.join(self.mriDir,mri_name))


            if (i == 0):
                image=npy_contents[...,:3]

            else:
                image=np.concatenate([image, npy_contents[...,:3]], 0)


            #im_temp = None
            #image_temp = None

        #image_all=np.transpose(image, (2, 0, 1,3))


        return image.astype('uint16')

