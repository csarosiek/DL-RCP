# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 16:18:14 2021

This script contains the dense block function for the DenseU-Net Code. 

@author: csarosiek
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import MaxPooling3D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

def DenseBlock4(channels, inputs):
    
    d11 = Conv2D(channels,(3,3), activation='relu',padding='same')(inputs)
    d11 = BatchNormalization(axis=3)(d11)
    d12 = Conv2D(channels,(3,3), activation='relu',padding='same')(d11)
    d12 = BatchNormalization(axis=3)(d12)
    
    d2 = concatenate([inputs,d12])
    d21 = Conv2D(channels,(3,3), activation='relu',padding='same')(d2)
    d21 = BatchNormalization(axis=3)(d21)
    d22 = Conv2D(channels,(3,3), activation='relu',padding='same')(d21)
    d22 = BatchNormalization(axis=3)(d22)
    
    d3 = concatenate([inputs,d12,d22])
    d31 = Conv2D(channels,(3,3), activation='relu',padding='same')(d3)
    d31 = BatchNormalization(axis=3)(d31)
    d32 = Conv2D(channels,(3,3), activation='relu',padding='same')(d31)
    d32 = BatchNormalization(axis=3)(d32)
    
    d4 = concatenate([inputs,d12,d22,d32])
    d41 = Conv2D(channels,(3,3), activation='relu',padding='same')(d4)
    d41 = BatchNormalization(axis=3)(d41)
    d42 = Conv2D(channels,(3,3), activation='relu',padding='same')(d41)
    d42 = BatchNormalization(axis=3)(d42)
    
    output = concatenate([inputs,d12,d22,d32,d42])
    
    return output

def DenseBlock2(channels, inputs):
    
    d11 = Conv2D(channels,(3,3), activation='relu',padding='same')(inputs)
    d11 = BatchNormalization(axis=3)(d11)
    d12 = Conv2D(channels,(3,3), activation='relu',padding='same')(d11)
    d12 = BatchNormalization(axis=3)(d12)
    
    d2 = concatenate([inputs,d12])
    d21 = Conv2D(channels,(3,3), activation='relu',padding='same')(d2)
    d21 = BatchNormalization(axis=3)(d21)
    d22 = Conv2D(channels,(3,3), activation='relu',padding='same')(d21)
    d22 = BatchNormalization(axis=3)(d22)
    
    output = concatenate([inputs,d12,d22])
    
    return output


def DenseBlockDropout(channels, inputs, DR):
    d11 = BatchNormalization(axis=3)(inputs)
    #d11 = Activation('relu')(d11)
    d11 = Conv2D(channels,(3,3), activation='relu',padding='same')(d11)
    d11 = Dropout(DR)(d11)
    d12 = BatchNormalization(axis=3)(d11)
    #d12 = Activation('relu')(d12)
    d12 = Conv2D(channels,(3,3), activation='relu',padding='same')(d12)
    d12 = Dropout(DR)(d12)
    
    d2 = concatenate([inputs,d12])
    d21 = BatchNormalization(axis=3)(d2)
    #d21 = Activation('relu')(d21)
    d21 = Conv2D(channels,(3,3), activation='relu',padding='same')(d21)
    d21 = Dropout(DR)(d21)
    d22 = BatchNormalization(axis=3)(d21)
    #d22 = Activation('relu')(d22)
    d22 = Conv2D(channels,(3,3), activation='relu',padding='same')(d22)
    d22 = Dropout(DR)(d22)
    
    d3 = concatenate([inputs,d12,d22])
    d31 = BatchNormalization(axis=3)(d3)
    #d31 = Activation('relu')(d31)
    d31 = Conv2D(channels,(3,3), activation='relu',padding='same')(d31)
    d31 = Dropout(DR)(d31)
    d32 = BatchNormalization(axis=3)(d31)
    #d32 = Activation('relu')(d32)
    d32 = Conv2D(channels,(3,3), activation='relu',padding='same')(d32)
    d32 = Dropout(DR)(d32)
    
    d4 = concatenate([inputs,d12,d22,d32])
    d41 = BatchNormalization(axis=3)(d4)
    #d41 = Activation('relu')(d41)
    d41 = Conv2D(channels,(3,3), activation='relu',padding='same')(d41)
    d41 = Dropout(DR)(d41)
    d42 = BatchNormalization(axis=3)(d41)
    #d42 = Activation('relu')(d42)
    d42 = Conv2D(channels,(3,3), activation='relu',padding='same')(d42)
    d42 = Dropout(DR)(d42)
    
    output = concatenate([inputs,d12,d22,d32,d42])
    
    return output

def DenseBlockDropout3D(channels, inputs, DR):
    d11 = BatchNormalization(axis=3)(inputs)
    #d11 = Activation('relu')(d11)
    d11 = Conv3D(channels,(3,3,3), activation='relu',padding='same')(d11)
    d11 = Dropout(DR)(d11)
    d12 = BatchNormalization(axis=3)(d11)
    #d12 = Activation('relu')(d12)
    d12 = Conv3D(channels,(3,3,3), activation='relu',padding='same')(d12)
    d12 = Dropout(DR)(d12)
    
    d2 = concatenate([inputs,d12])
    d21 = BatchNormalization(axis=3)(d2)
    #d21 = Activation('relu')(d21)
    d21 = Conv3D(channels,(3,3,3), activation='relu',padding='same')(d21)
    d21 = Dropout(DR)(d21)
    d22 = BatchNormalization(axis=3)(d21)
    #d22 = Activation('relu')(d22)
    d22 = Conv3D(channels,(3,3,3), activation='relu',padding='same')(d22)
    d22 = Dropout(DR)(d22)
    
    d3 = concatenate([inputs,d12,d22])
    d31 = BatchNormalization(axis=3)(d3)
    #d31 = Activation('relu')(d31)
    d31 = Conv3D(channels,(3,3,3), activation='relu',padding='same')(d31)
    d31 = Dropout(DR)(d31)
    d32 = BatchNormalization(axis=3)(d31)
    #d32 = Activation('relu')(d32)
    d32 = Conv3D(channels,(3,3,3), activation='relu',padding='same')(d32)
    d32 = Dropout(DR)(d32)
    
    d4 = concatenate([inputs,d12,d22,d32])
    d41 = BatchNormalization(axis=3)(d4)
    #d41 = Activation('relu')(d41)
    d41 = Conv3D(channels,(3,3,3), activation='relu',padding='same')(d41)
    d41 = Dropout(DR)(d41)
    d42 = BatchNormalization(axis=3)(d41)
    #d42 = Activation('relu')(d42)
    d42 = Conv3D(channels,(3,3,3), activation='relu',padding='same')(d42)
    d42 = Dropout(DR)(d42)
    
    output = concatenate([inputs,d12,d22,d32,d42])
    
    return output

def DenseBlockDropout2(channels, inputs, DR):
    d11 = BatchNormalization(axis=3)(inputs)
    d11 = Activation('relu')(d11)
    d11 = Conv2D(channels,(3,3), activation='relu',padding='same')(d11)
    d11 = Dropout(DR)(d11)
    d12 = BatchNormalization(axis=3)(d11)
    d12 = Activation('relu')(d12)
    d12 = Conv2D(channels,(3,3), activation='relu',padding='same')(d12)
    d12 = Dropout(DR)(d12)
    
    d2 = concatenate([inputs,d12])
    d21 = BatchNormalization(axis=3)(d2)
    d21 = Activation('relu')(d21)
    d21 = Conv2D(channels,(3,3), activation='relu',padding='same')(d21)
    d21 = Dropout(DR)(d21)
    d22 = BatchNormalization(axis=3)(d21)
    d22 = Activation('relu')(d22)
    d22 = Conv2D(channels,(3,3), activation='relu',padding='same')(d22)
    d22 = Dropout(DR)(d22)
    
    output = concatenate([inputs,d12,d22])
    
    return output

def TransitionDown(channels, inputs):
    t11 = BatchNormalization(channels)(inputs)
    t12 = Activation('relu')(t11)
    t13 = Conv2D(channels, (3,3),padding='same')(t12)
    t14 = Dropout(0.2)(t13)
    
    output = MaxPooling2D(pool_size=(2,2))(t14)
    
    return output

def TransitionUp(channels, inputs):
    u11 = BatchNormalization(channels)(inputs)
    u12 = Activation('relu')(u11)
    u13 = Conv2DTranspose(channels/2, (2,2), stride=(2,2), padding='same')(u12)
    output = Dropout(0.2)(u13)    
    
    return output