# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 11:02:34 2022

Dense UNet Correction
This script contains all the functions necessary to apply the dense unet 2D
ACC method to the contours.

@author: csarosiek
"""

import numpy as np
from tensorflow.keras.models import load_model
import cv2
import os
import scipy.io as sio
import matplotlib.pyplot as plt


def ModelLoad(organ):
    path = './data/FinalWorkflowModels/'
    os.environ["CUDA_VISIBLE_DEVICES"]= '-1'
    #print(os.path.isfile(path+'dunet_minor_'+organ+'.h5'))
    try:
        minorDU = load_model(path+'ACC_'+organ+'_Minor.h5',compile=False)
    except:
        print('No minor Model')
        minorDU = None
    try:
        majorDU = load_model(path+'ACC_'+organ+'_Major.h5',compile=False)
    except:
        print('No major model')
        majorDU = None

    return minorDU, majorDU

def IntensityClip(array,percentile):
    hist = np.histogram(np.ndarray.flatten(array),bins=200)
    numbelow = sum(hist[0])*percentile
    check = 0
    k = 2
    while k < len(hist[0])-1:
        check = sum(hist[0][:k])
        if check > numbelow:
            break
        else:
            k += 1
    maximum = hist[1][k]

    array = np.clip(array,a_min=None,a_max=maximum)
    return array


def Normalize(mr, value):
    MR = np.zeros(mr.shape)
    minimum = np.min(mr[mr>0])
    maximum =  np.max(mr)
    i = 0
    while i < mr.shape[0]:
        j = 0
        while j < mr.shape[1]:
            if int(mr[i,j]) > 0:
                MR[i,j] = int(value*(mr[i,j] - minimum)/np.absolute(maximum - minimum))
            j += 1
        i += 1
    #MR = ((mr - minimum)/np.absolute(maximum - minimum))*value
    return MR



def PrepareDUSlice_train(imageslice, contourslice, gtslice, organ, pixelW, fname, pad=20, ImageSize=64):
    #if 'Bowel' in organ:
    #    contourslice = contourslice[50:200,:]
    #    pad0 = pad+50
    #else:
    #    pad0 = pad

    Slicelim = np.where(contourslice)
    lim1_max = min([max(Slicelim[0])+pad, contourslice.shape[0]])
    lim1_min = max([min(Slicelim[0])-pad, 0])
    lim2_max = min([max(Slicelim[1])+pad, contourslice.shape[1]])
    lim2_min = max([min(Slicelim[1])-pad, 0])
    limits = [lim1_min, lim1_max, lim2_min, lim2_max]
    #if 'Bowel' in organ:
    #    limits = [lim1_min+50,lim1_max+50, lim2_min,lim2_max]

    ## Crop the slice
    mrcrop_MR = imageslice[lim1_min:lim1_max,lim2_min:lim2_max]
    mrcrop_DL = contourslice[lim1_min:lim1_max,lim2_min:lim2_max]
    mrcrop_GT = gtslice[lim1_min:lim1_max,lim2_min:lim2_max]


    matout = {'MR':mrcrop_MR,'GT':mrcrop_GT,'DL':mrcrop_DL,'pixel':pixelW}
    sio.savemat(fname, matout)

    ## Resize the image
    im = np.zeros([1,ImageSize,ImageSize,3])
    im[0,:,:,0] = cv2.resize(mrcrop_MR,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_LINEAR) ##bilinear interpolation?
    im[0,:,:,1] = cv2.resize(mrcrop_DL,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST)
    im[0,:,:,2] = cv2.resize(mrcrop_GT,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST)

    im[0,:,:,0] = cv2.GaussianBlur(im[0,:,:,0],[3,3],1)
    #print('DU Slice Prepared')
    return im, limits

def PrepareDUSlice_train_multiorgan(imageslice, contourslice, gtslice, pixelW, fname, pad=20, ImageSize=64):
    #if 'Bowel' in organ:
    #    contourslice = contourslice[50:200,:]
    #    pad0 = pad+50
    #else:
    #    pad0 = pad

    Slicelim = np.where(contourslice)
    lim1_max = min([max(Slicelim[0])+pad, contourslice.shape[0]])
    lim1_min = max([min(Slicelim[0])-pad, 0])
    lim2_max = min([max(Slicelim[1])+pad, contourslice.shape[1]])
    lim2_min = max([min(Slicelim[1])-pad, 0])
    limits = [lim1_min, lim1_max, lim2_min, lim2_max]
    #if 'Bowel' in organ:
    #    limits = [lim1_min+50,lim1_max+50, lim2_min,lim2_max]

    ## Crop the slice
    mrcrop_MR = imageslice[lim1_min:lim1_max,lim2_min:lim2_max]
    mrcrop_DL = contourslice[lim1_min:lim1_max,lim2_min:lim2_max]
    mrcrop_GT = gtslice[lim1_min:lim1_max,lim2_min:lim2_max]


    #matout = {'MR':mrcrop_MR,'GT':mrcrop_GT,'DL':mrcrop_DL,'pixel':pixelW}
    #sio.savemat(fname, matout)

    ## Resize the image
    im = np.zeros([1,ImageSize,ImageSize,3])
    im[0,:,:,0] = cv2.resize(mrcrop_MR,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_LINEAR) ##bilinear interpolation?
    im[0,:,:,1] = cv2.resize(mrcrop_DL,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST)
    im[0,:,:,2] = cv2.resize(mrcrop_GT,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST)

    #im[0,:,:,0] = cv2.GaussianBlur(im[0,:,:,0],[3,3],1)
    #print('DU Slice Prepared')
    return im, limits


def PrepareDUSlice_multiorgan(imageslice, contourslice, pad=20, ImageSize=64):

    Slicelim = np.where(contourslice)
    lim1_max = min([max(Slicelim[0])+pad, contourslice.shape[0]])
    lim1_min = max([min(Slicelim[0])-pad, 0])
    lim2_max = min([max(Slicelim[1])+pad, contourslice.shape[1]])
    lim2_min = max([min(Slicelim[1])-pad, 0])
    limits = [lim1_min, lim1_max, lim2_min, lim2_max]
    #if 'Bowel' in organ:
    #    limits = [lim1_min+50,lim1_max+50, lim2_min,lim2_max]

    ## Crop the slice
    mrcrop_MR = imageslice[lim1_min:lim1_max,lim2_min:lim2_max]
    mrcrop_DL = contourslice[lim1_min:lim1_max,lim2_min:lim2_max]
    #mrcrop_GT = gtslice[lim1_min:lim1_max,lim2_min:lim2_max]

    ## Resize the image
    im = np.zeros([1,ImageSize,ImageSize,2])
    im[0,:,:,0] = cv2.resize(mrcrop_MR,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_LINEAR) ##bilinear interpolation?
    im[0,:,:,1] = cv2.resize(mrcrop_DL,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST)
    #im[0,:,:,2] = cv2.resize(mrcrop_GT,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST)

    #im[0,:,:,0] = cv2.GaussianBlur(im[0,:,:,0],[3,3],1)
    #print('DU Slice Prepared')
    return im, limits


def PrepareDUSlice(imageslice, contourslice, organ, pad=20, ImageSize=64):
    #if 'Bowel' in organ:
    #    contourslice = contourslice[50:200,:]
    #    pad0 = pad+50
    #else:
    #    pad0 = pad

    Slicelim = np.where(contourslice)
    lim1_max = min([max(Slicelim[0])+pad, contourslice.shape[0]])
    lim1_min = max([min(Slicelim[0])-pad, 0])
    lim2_max = min([max(Slicelim[1])+pad, contourslice.shape[1]])
    lim2_min = max([min(Slicelim[1])-pad, 0])
    limits = [lim1_min, lim1_max, lim2_min, lim2_max]
    #if 'Bowel' in organ:
    #    limits = [lim1_min+50,lim1_max+50, lim2_min,lim2_max]

    ## Crop the slice
    mrcrop_MR = imageslice[lim1_min:lim1_max,lim2_min:lim2_max]
    mrcrop_DL = contourslice[lim1_min:lim1_max,lim2_min:lim2_max]

    ## Resize the image
    im = np.zeros([1,ImageSize,ImageSize,2])
    im[0,:,:,0] = cv2.resize(mrcrop_MR,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_LINEAR) ##bilinear interpolation?
    im[0,:,:,1] = cv2.resize(mrcrop_DL,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST)

    im[0,:,:,0] = cv2.GaussianBlur(im[0,:,:,0],[3,3],1)
    #print('DU Slice Prepared')
    return im, limits

def DUprediction(data, organ, category, minorDU, majorDU):
    data = data.astype('float32')

    # if 'Duo' in organ or 'mall' in organ:
    #     data = (data-np.min(data))/(np.absolute(np.max(data)-np.min(data)))
    # else:
    #     data[0,:,:,0] = (data[0,:,:,0]-np.min(data[0,:,:,0]))/(np.absolute(np.max(data[0,:,:,0])-np.min(data[0,:,:,0])))

    data[0,:,:,0] = (data[0,:,:,0]-np.min(data[0,:,:,0]))/(np.absolute(np.max(data[0,:,:,0])-np.min(data[0,:,:,0])))
    data[0,:,:,1] = (data[0,:,:,1]-np.min(data[0,:,:,1]))/(np.absolute(np.max(data[0,:,:,1])-np.min(data[0,:,:,1])))

    if category == 2:
        DLCorr=minorDU.predict_on_batch(data)
    elif category == 3:
        DLCorr=majorDU.predict_on_batch(data)
    #print('DU predict')
    return DLCorr


def DUprediction_multiorgan(data, model):
    data = data.astype('float16')

    data[0,:,:,0] = (data[0,:,:,0]-np.min(data[0,:,:,0]))/(np.absolute(np.max(data[0,:,:,0])-np.min(data[0,:,:,0])))
    #data[0,:,:,1] = (data[0,:,:,1]-np.min(data[0,:,:,1]))/(np.absolute(np.max(data[0,:,:,1])-np.min(data[0,:,:,1])))

    DLCorr=model.predict_on_batch(data)
    #print('DU predict')
    return DLCorr


def UpSize(DLCorr, originalcontour, limits, thresh=0.5):
    extent0 = limits[1] - limits[0]
    extent1 = limits[3] - limits[2]

    ## Resize back to original size
    DLcorr_resize = cv2.resize(DLCorr[0,:,:,1],dsize=(extent1,extent0),interpolation=cv2.INTER_CUBIC)
    t = np.ones(DLcorr_resize.shape)*thresh
    DLcorr_resize_mask = np.greater_equal(DLcorr_resize, t)

    ## Combine back into full slice and save as npy format
    corr_slice = originalcontour
    #plt.imshow(corr_slice)
    #print(limits)
    corr_slice[limits[0]:limits[1],limits[2]:limits[3]] = DLcorr_resize_mask
    #plt.contour(corr_slice,0,colors='black')
    #plt.show()
    #print('upsize complete')
    return corr_slice

def UpSize_multiorgan(DLCorr, originalcontour, limits, thresh=0.5):
    extent0 = limits[1] - limits[0]
    extent1 = limits[3] - limits[2]

    ACC_mask = np.zeros((extent0,extent1))
    ## Resize back to original size
    o = 1
    while o < DLCorr.shape[3]:
        DLcorr_resize = cv2.resize(DLCorr[0,:,:,o],dsize=(extent1,extent0),interpolation=cv2.INTER_CUBIC)
        t = np.ones(DLcorr_resize.shape)*thresh
        DLCorr_resize_mask = np.greater_equal(DLcorr_resize, t)

        DLCorr_organ = np.greater_equal(DLCorr_resize_mask,thresh)
        ACC_mask = ACC_mask + DLCorr_organ*o

        o += 1

    ## Combine back into full slice and save as npy format
    corr_slice = originalcontour
    #plt.imshow(corr_slice)
    corr_slice[limits[0]:limits[1],limits[2]:limits[3]] = ACC_mask

    #plt.contour(corr_slice,0,colors='black')
    #plt.show()
    #print('upsize complete')
    return corr_slice


def ApplyDenseUNetACC(MRvolume,contourvolume,organ,categories):
    print('Dense UNet Start')
    ## Load the models for that organ
    minorDU = None
    majorDU = None
    minorDU, majorDU = ModelLoad(organ)
    print('DU Models loaded')

    ##Setup New Contour Volume
    ACCcontourVolume = contourvolume.astype(int)

    ##Volumetric Preprocessing:
    #MRvolume = MRvolume - np.min(MRvolume)

    i = 0
    while i < MRvolume.shape[2]:
        MRslice = MRvolume[:,:,i]
        category = categories[-1*i-1]
        DLContour = contourvolume[:,:,-1*i-1].astype(int)
        # plt.imshow(MRslice,cmap='gray')
        # plt.contour(DLContour,0,colors='green')
        # plt.title(str(category))
        # plt.show()

        if np.max(DLContour) == 0:
            i += 1
            continue
        #print(category)
        if category != 2 and category!=3:
            #skip any slice that doesn't require minor/major edits
            #print('skipping acceptable')
            i += 1
            continue
        elif category == 2 and minorDU == None:
            #skip any slice that we don't have a minor model for
            print('skipping minor')
            i += 1
            continue
        elif category == 3 and majorDU == None:
            #skip any slice that we don't have a major model for
            print('skipping major')
            i += 1
            continue
        else:
            #print('correcting slice')
            MRslice = MRslice - np.min(MRslice)
            #MRslice = IntensityClip(MRslice,0.99)
            #MRslice = Normalize(MRslice,1200)

            if organ == 'Duodenum' and category == 3:
                PAD = 75
                IMAGESIZE = 128
            else:
                PAD = 40
                IMAGESIZE = 64

            # print(organ)
            # print(category)
            # print(PAD)
            # print(IMAGESIZE)
            data, limits = PrepareDUSlice(MRslice, DLContour, organ, pad=PAD, ImageSize=IMAGESIZE)
            DLCorr = DUprediction(data, organ, category, minorDU, majorDU)

            #print(np.max(DLCorr[0,:,:,1]))

            # plt.imshow(DLCorr[0,:,:,1])
            # plt.contour(data[0,:,:,1],0,colors='black')
            # plt.show()

            ACCcontour = UpSize(DLCorr,DLContour,limits, thresh=0.5)
            #if np.max(ACCcontour) == 0:
                #print('ACC deleted contour')
                #ACCcontour = DLContour
            #except:
            #    print('DenseUNet ACC failed on slice')
            #    ACCcontour = DLContour
        #if 'owel' in organ:
        #    ACCcontour = ACCcontour + DLContour
        #    ACCcontour[ACCcontour>0] = 1

        #ACCcontour = cv2.medianBlur(ACCcontour.astype('float32'),5)

        # plt.imshow(MRslice,cmap='gray')
        # plt.contour(DLContour,1,colors='green')
        # plt.contour(ACCcontour,0,colors='red')
        # #plt.imshow(ACCcontour - DLContour, vmin = -1, vmax = 1)
        # plt.title(str(category))
        # plt.show()

        #print(np.max(ACCcontour-DLContour), np.min(ACCcontour-DLContour))
        ACCcontourVolume[:,:,-1*i-1] = ACCcontour
        i += 1

    # i = 0
    # while i < ACCcontourVolume.shape[2]:
    #     A = ACCcontourVolume[:,:,i]
    #     A = cv2.medianBlur(A.astype('float32'),5)
    #     ACCcontourVolume[:,:,i] = A
    #     i += 1
    ACCcontourVolume = ACCcontourVolume.astype(bool)
    print('2D ACC complete')
    return ACCcontourVolume



def ApplyDenseUNetACC_multiorgan(MRvolume,contourvolume):
    print('Dense UNet Start')
    path = './multiorgan_data/organs_7/model_optimal/'
    multiorganmodel = None
    multiorganmodel = load_model(path+'ACC_multiorgan.h5',compile=False)
    print('DU Model loaded')

    ##Setup New Contour Volume
    ACCcontourVolume = contourvolume.astype(int)

    ##Volumetric Preprocessing:
    #MRvolume = MRvolume - np.min(MRvolume)

    i = 0
    while i < MRvolume.shape[2]:
        MRslice = MRvolume[:,:,i]
        DLContour = contourvolume[:,:,-1*i-1].astype(int)
        if np.max(DLContour) == 0:
            i += 1
            continue

        MRslice = MRslice - np.min(MRslice)
        #try:
        data, limits = PrepareDUSlice_multiorgan(MRslice, DLContour, pad=40, ImageSize=128)
        DLCorr = DUprediction_multiorgan(data, multiorganmodel)
        #np.save('./shape'+str(i)+'.npy',DLCorr)

        ACCcontour = UpSize_multiorgan(DLCorr, DLContour, limits, thresh=0.5)

        ACCcontourVolume[:,:,-1*i-1] = ACCcontour
        i += 1

    ACCcontourVolume = ACCcontourVolume
    print('2D ACC complete')
    return ACCcontourVolume

##EOF
