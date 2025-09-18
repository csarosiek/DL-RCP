# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 15:21:37 2024

@author: csarosiek
"""

import numpy as np
import cv2
import os
import pydicom
#from tools import data
import numpy as np
import denseunet
import contourdata
import scipy.io as sio
import matplotlib.pyplot as plt
import multiprocessing

init = 'RR_CT'
GT_path = 'G:/Physicist/people/DLAS/DICOM/MRL/BTFE_F_DELREC/training/orig/GT/'
Init_path = 'G:/Physicist/people/DLAS/DICOM/MRL/BTFE_F_DELREC/training/orig/'+init
patients = os.listdir(GT_path)

data_dir = 'G:/Physicist/people/Sarosiek/1_DenseUNet_DelRec/multiorgan_data/organs_7/training_crop256_resize128/'+init

try:
    os.mkdir(data_dir)
except:
    pass


organlist = ['Duodenum','Stomach','Colon','Bowel_Small','Liver','Kidney_R','Kidney_L']#['Duodenum','Stomach','Colon','Bowel_Small']
c = 0
# for organID in organlist:
#     #Create folders for data
try:
    os.mkdir(os.path.join(data_dir))
except:
    pass

try:
    os.mkdir(os.path.join(data_dir,'resize-npy'))
except:
    pass

try:
    os.mkdir(os.path.join(data_dir,'cropped-mat'))
except:
    pass

#for pat in patients:

def make_data(p):
    pat = patients[p]
    #print(pat)
    struct_path = os.path.join(Init_path,pat)
    image_path = os.path.join(GT_path,pat)
    gtstruct_path = os.path.join(GT_path,pat)

    ## get images and organ-specific contour masks from the DICOM files
    MRvolume, contourvolume, pixelW = contourdata.get_DL_data_multiorgan(struct_path,image_path,organ_list=organlist)
    dicom_images2, dicom_gt_masks, pixelW2 = contourdata.get_DL_data_multiorgan(gtstruct_path,image_path,organ_list=organlist)

    # plt.imshow(contourvolume[:,:,40])
    # plt.show()

    # plt.imshow(dicom_gt_masks[:,:,40])
    # plt.show()


    if dicom_gt_masks == [] or contourvolume == []:
        print('No organs in patient')
        i = 999
    else:
        i = 0

    ##Volumetric Preprocessing:
    print(np.max(MRvolume))
    #MRvolume = denseunet.IntensityClip(MRvolume,90)
    thresh =  np.percentile(MRvolume,99.5)
    MRvolume[MRvolume>thresh] = thresh
    x,y,s = MRvolume.shape
    startx = x//2 - 256//2
    starty = y//2 - 256//2
    MRvolume = MRvolume[startx:startx+256, starty:starty+256,:].astype('uint16')
    contourvolume = contourvolume[startx:startx+256, starty:starty+256,:].astype('uint8')
    dicom_gt_masks = dicom_gt_masks[startx:startx+256, starty:starty+256,:].astype('uint8')
    print(np.max(MRvolume))

    #i = 0
    while i < MRvolume.shape[2]:
        MRslice = MRvolume[:,:,i]
        #category = categories[i]
        DLContour = contourvolume[:,:,-1*i-1].astype(int)
        GTContour = dicom_gt_masks[:,:,-1*i-1].astype(int)
        if np.max(DLContour)==0:
            i+=1
            continue

        # if i%40 == 0:
        #     plt.imshow(MRslice,cmap='gray')
        #     plt.contour(GTContour,[0,1,2,3,4,5,6,7],cmap='tab10')
        #     plt.contour(DLContour,[0,1,2,3,4,5,6,7],cmap='tab10')
        #     plt.show()

        #MRslice = MRslice - np.min(MRslice)
        #MRslice = denseunet.IntensityClip(MRslice,0.99)
        #MRslice = denseunet.Normalize(MRslice,1200)
        fname = data_dir+'/cropped-mat/'+pat+'_'+init+'_'+str(i).zfill(3)+'.mat'
        matout = {'MR':MRslice,'GT':GTContour,'DL':DLContour,'pixel':pixelW}
        sio.savemat(fname, matout)

        ## Resize the image
        ImageSize = 128
        im = np.zeros([1,ImageSize,ImageSize,3])
        im[0,:,:,0] = cv2.resize(MRslice,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_AREA).astype('uint16') ##bilinear interpolation?
        im[0,:,:,1] = cv2.resize(DLContour,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST).astype('uint8')
        im[0,:,:,2] = cv2.resize(GTContour,dsize=(ImageSize,ImageSize),interpolation=cv2.INTER_NEAREST).astype('uint8')

        # if i%40 == 0:
        #     plt.imshow(data[0,:,:,0],cmap='gray')
        #     plt.contour(data[0,:,:,1],[0,1,2,3,4,5,6,7],cmap='tab10')
        #     plt.contour(data[0,:,:,2],[0,1,2,3,4,5,6,7],cmap='tab10')
        #     plt.show()

        npyfile = data_dir+'/resize-npy/'+pat+'_'+init+'_'+str(i).zfill(3)+'.npy'
        np.save(npyfile,im)

        i+=1
    return print(pat)



if __name__ == '__main__':
    with multiprocessing.Pool(5) as pool:
        pool.map(make_data, range(len(patients)))

    #print(results)
