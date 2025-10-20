#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:47:12 2019

DL-RCP Model Training

Required Files to run:
    LoadData_v3_aug.py
    DataGenerator_v3_aug.py
    DenseLayers.py

@author: csarosiek
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

from LoadData_v4_python_aug import Dataloader
from DataGenerator_v4_multiorgan import DataGenerator
import DenseLayers
#import DenseUnetMetrics as DenseMetrics
#import PrepareData_v2

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils
from tensorflow.keras.metrics import TruePositives, FalsePositives, TrueNegatives, FalseNegatives
from tensorflow.keras.metrics import CategoricalAccuracy, SparseCategoricalAccuracy, TopKCategoricalAccuracy
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError, Recall, Precision
from tensorflow_addons.metrics import F1Score
from tensorflow.keras.losses import SparseCategoricalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.losses import MeanSquaredError as mseloss
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow_addons.losses import SigmoidFocalCrossEntropy

from tensorflow.keras.preprocessing.image import ImageDataGenerator



#launch=os.getcwd()
#os.environ["CUDA_PATH"]=''
#os.environ["CUDA_PATH_V11_4"]=''
os.environ["CUDA_VISIBLE_DEVICES"]='-1' # Change to find the GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

## Define the hyperparameters here:
NUM_CHANNELS = 8 
BATCH_SIZE = 16  # Can adjust
VAL_BATCH = BATCH_SIZE
IMG_HEIGHT = 128 # Defined in the Preprocessing Code
IMG_WIDTH = IMG_HEIGHT
KERNEL_SIZE = 3
NUM_CLASSES = 8 # Number of organs plus 1 for background (8 = 7 organs + 1 background)
DR = 0.0 ## DROPOUT RATE
EPOCHS = 1000 ## Number of epochs to train for
LR = 5e-5 ## Learning Rate

## Define the directories here:
folder = 'G:/Physicist/people/Sarosiek/1_DenseUNet_DelRec/multiorgan_data/organs_7/training_pad40_resize128/'
train_path = folder+'/training/'
test_path = folder+'/validation/'
log_path = folder+'/temp-'+time.strftime("%Y%m%d-%H%M%S")+'/'


# load all dataset
print('Load training set')
training = Dataloader(train_path)
print('Load validation set')
validation = Dataloader(test_path)

# data generator
#training_generator = DataGenerator(training, BATCH_SIZE, NUM_CLASSES,shuffle=True)
validation_generator = DataGenerator(validation, VAL_BATCH, NUM_CLASSES,shuffle=True)


# training preparation and augmentation


training_x = np.empty((training.image_all.shape[0], training.image_all.shape[1], training.image_all.shape[2], 2))
training_y = np.empty((training.image_all.shape[0], training.image_all.shape[1], training.image_all.shape[2], 1))

training_x=training.image_all[:,:,:,0:2] #Slice data and DL contour
training_y=training.image_all[:,:,:,2] #Good contour


training_y[training_y > 7] = 0

training_x=training_x.astype('float32')
training_y=training_y.astype('int32')
training_y=utils.to_categorical(training_y)

#     #normalization
for i in range(training_x.shape[0]):
    #training_x[i] = (training_x[i]-np.min(training_x[i]))/(np.absolute(np.max(training_x[i])-np.min(training_x[i])))
    training_x[i,:,:,0] = (training_x[i,:,:,0]-np.min(training_x[i,:,:,0]))/(np.absolute(np.max(training_x[i,:,:,0])-np.min(training_x[i,:,:,0])))
    #training_x[i,:,:,1] = (training_x[i,:,:,1]-np.min(training_x[i,:,:,1]))/(np.absolute(np.max(training_x[i,:,:,1])-np.min(training_x[i,:,:,1])))


data_gen_args = dict(rotation_range=45,
                    zoom_range=0.3,
                    horizontal_flip=False,
                    vertical_flip=False,
                    height_shift_range=0.3,
                    width_shift_range=0.3
                  )

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

seed = 1


image_generator = image_datagen.flow(training_x, seed=seed, batch_size=BATCH_SIZE, shuffle=True)
mask_generator = mask_datagen.flow(training_y, seed=seed, batch_size=BATCH_SIZE, shuffle=True)

train_generator = zip(image_generator, mask_generator)




#os.chdir(launch)
#inputs = Input(( None, None, NUM_CHANNELS))
inputs = Input((IMG_HEIGHT,IMG_WIDTH,2))#NUM_CHANNELS))
print(inputs.shape)

chan = 64
model = Sequential()
#Layer 1D
c1 = Conv2D(chan, (3, 3), activation='relu',  padding='same') (inputs)
p1 = MaxPooling2D(pool_size=(2, 2)) (c1)

#Dense Block
p1d = DenseLayers.DenseBlockDropout(chan*2, p1, DR)
#p1d = DenseLayers.DenseBlock4(chan*2, p1)

#Layer 2D
c2 = BatchNormalization(axis=3) (p1d)
c2 = Conv2D(chan*2, (3, 3), activation='relu',  padding='same') (c2)
c2 = Dropout(DR)(c2)
p2 = MaxPooling2D(pool_size=(2, 2)) (c2)

#Dense Block
p2d = DenseLayers.DenseBlockDropout(chan*4, p2, DR)
#p2d = DenseLayers.DenseBlock4(chan*4, p2)

#Layer 3D
c3 = BatchNormalization(axis=3) (p2d)
c3 = Conv2D(chan*4, (3, 3), activation='relu',  padding='same') (c3)
c3 = Dropout(DR)(c3)
p3 = MaxPooling2D(pool_size=(2, 2)) (c3)

#Dense Block
p3d = DenseLayers.DenseBlockDropout(chan*8, p3, DR)
#p3d = DenseLayers.DenseBlock4(chan*8, p3)

#Layer 4D
c4 = BatchNormalization(axis=3) (p3d)
c4 = Conv2D(chan*8, (3, 3), activation='relu',  padding='same') (c4)
c4 = Dropout(DR)(c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

#Cross layer
c5 = BatchNormalization(axis=3) (p4)
c5 = Conv2D(chan*16, (3, 3), activation='relu',  padding='same') (c5)

#Layer 4U
u6 = BatchNormalization(axis=3) (c5)
u6 = Conv2DTranspose(chan*8, (2, 2), strides=(2, 2), activation='relu',padding='same') (u6)
u6 = Dropout(DR)(u6)
c6 = concatenate([u6, c4]) ## Long skip connection

#Dense Block
#c6d = DenseLayers.DenseBlock4(chan*8, c6)
c6d = DenseLayers.DenseBlockDropout(chan*8, c6, DR)

#Layer 3U
u7 = BatchNormalization(axis=3) (c6d)
u7 = Conv2DTranspose(chan*4, (2, 2), strides=(2, 2), padding='same', activation='relu') (u7)
u7 = Dropout(DR)(u7)
c7 = concatenate([u7, c3]) ## Long skip connection

#Dense Block
c7d = DenseLayers.DenseBlockDropout(chan*4, c7, DR)
#c7d = DenseLayers.DenseBlock4(chan*4, c7)

#Layer 2U
u8 = BatchNormalization(axis=3) (c7d)
u8 = Conv2DTranspose(chan*2, (2, 2), strides=(2, 2), padding='same',activation='relu') (u8)
u8 = Dropout(DR)(u8)
c8 = concatenate([u8, c2]) ## Long skip connection

#Dense Block
c8d = DenseLayers.DenseBlockDropout(chan*2, c8, DR)
#c8d = DenseLayers.DenseBlock4(chan*2, c8)

#Layer 1U
u9 = BatchNormalization(axis=3) (c8d)
u9 = Conv2DTranspose(chan, (2, 2), strides=(2, 2), padding='same',activation='relu') (u9)
u9 = Dropout(DR)(u9)
c9 = concatenate([u9, c1], axis=3) ## Long skip connection


outputs = Conv2D(NUM_CLASSES, (1, 1), activation='softmax') (c9) #softmax? sigmoid

model = Model(inputs=[inputs], outputs=[outputs])

model.compile(optimizer=Adam(LR),
              loss=CategoricalCrossentropy(), #
              metrics=['categorical_accuracy',TopKCategoricalAccuracy(k=5),
                       Recall(class_id=1,name='recall_1'),Recall(class_id=2,name='recall_2'),Recall(class_id=3,name='recall_3'),Recall(class_id=4,name='recall_4'),
                       Precision(class_id=1,name='precision_1'),Precision(class_id=2,name='precision_2'),Precision(class_id=3,name='precision_3'),Precision(class_id=4,name='precision_4')])
#model.summary()


tensorboard = TensorBoard(log_dir=log_path, histogram_freq=0,
                          write_graph=True, write_images=False)
                          #/0218_ep1000_bs16_8_9_rot90_zoom3/
tensorboard.set_model(model)
checkpoint = ModelCheckpoint(log_path+'/d-unet_model.h5', verbose=1,
                             monitor='val_loss',save_best_only=True, mode='auto')
# checkpoint2 = ModelCheckpoint(log_path+'/d-unet_model-val_recall.h5', verbose=1,
#                              monitor='val_recall_1',save_best_only=True, mode='max')
# checkpoint3 = ModelCheckpoint(log_path+'/d-unet_model-val_precision.h5', verbose=1,
#                              monitor='val_precision_1',save_best_only=True, mode='max')
#earlystop = EarlyStopping(monitor='loss',patience=50)
checkpoint.set_model(model)

# with open(log_path + 'd-unet-modelsummary.txt','w+') as fh:
#     model.summary(print_fn=lambda x: fh.write(x + '\n'))

#results = model.fit_generator(train_generator, steps_per_epoch=(len(training_x) // BATCH_SIZE), epochs=10, validation_data=validation_generator, use_multiprocessing=False,workers=6,callbacks=[tensorboard, checkpoint])
results = model.fit(train_generator, steps_per_epoch=(len(training_x) // BATCH_SIZE), epochs=EPOCHS, validation_data=validation_generator, use_multiprocessing=False,workers=6,callbacks=[tensorboard, checkpoint])


# plt.plot(epochs,loss,'-',label='loss')
# plt.plot(epochs,val_loss,'-',label='val_loss')
# plt.legend()
# plt.title('Model Loss per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim(0,1)
# plt.savefig(log_path+'/d-unet_model_loss.png', bbox_inches='tight',dpi=600)
# plt.show()

# plt.plot(epochs, F1, '-',label='F1')
# plt.plot(epochs,val_F1,'-',label='Validation F1')
# plt.legend()
# plt.title('F1 (DSC) per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('F1 (DSC)')
# plt.ylim(0,1)
# plt.savefig(log_path+'/d-unet_model_F1.png', bbox_inches='tight',dpi=600)
# plt.show()

# plt.plot(epochs,bin_acc,'-', label='Binary_accuracy')
# plt.plot(epochs,val_bin_acc,'-',label='Validation Binary_accuracy')
# plt.legend()
# plt.title('Binary Accuracy per Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Binary Accuracy')
# plt.ylim(0,1)
# plt.savefig(log_path+'/d-unet_model_bin_acc.png', bbox_inches='tight',dpi=600)
# plt.show()


## Print training results to a csv file. 
df = pd.DataFrame(results.history)
df.to_csv(log_path+'/d-unet_model_history.csv')




### eof

