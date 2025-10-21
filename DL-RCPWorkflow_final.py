# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:42:23 2023
DL-RCP Production Code. Reads DICOM files and produces a new RTStructure Set with the updated contours. 

@author: csarosiek
"""

import os
import numpy as np
import contourdata
import denseunet
from rt_utils import RTStructBuilder
import scipy.io as sio
import time

###


def testACC(contour_mask):
    ## We'll just add a giant rectangle to the intercepted file.
    size = contour_mask.shape
    for slices in contour_mask:
        slices[size[1]//5:2*(size[1]//5),size[2]//5:2*(size[2]//5)] = 1

    return contour_mask

def remove_old_ROI(rtstruct, organID):
    roi_contour_sequence = rtstruct.ds.ROIContourSequence
    structure_set_roi_sequence = rtstruct.ds.StructureSetROISequence

    # Find the index of the ROI to remove
    index_to_old = None
    for i, roi_entry in enumerate(structure_set_roi_sequence):
        if roi_entry.ROIName == organID:
            index_to_old = i
            break

    # Find the index of the ROI to remove
    index_to_acc = None
    for i, roi_entry in enumerate(structure_set_roi_sequence):
        if roi_entry.ROIName == organID+'ACC':
            index_to_acc = i
            break

    # Remove the ROI from the sequences
    if index_to_old is not None and index_to_acc is not None :
        roi_contour_sequence[index_to_old].ContourSequence=roi_contour_sequence[index_to_acc].ContourSequence

        # Remove the acc from ROIContourSequence
        roi_contour_sequence.pop(index_to_acc)
        # Remove the corresponding entry from StructureSetROISequence
        structure_set_roi_sequence.pop(index_to_acc)

def remove_all_ROI(rtstruct):
    roi_contour_sequence = rtstruct.ds.ROIContourSequence
    structure_set_roi_sequence = rtstruct.ds.StructureSetROISequence

    i = 0
    while i < len(structure_set_roi_sequence):

        # Remove the acc from ROIContourSequence
        roi_contour_sequence.pop(i)
        # Remove the corresponding entry from StructureSetROISequence
        structure_set_roi_sequence.pop(i)
        i += 1


def dice_coef(y_true, y_pred, thresh):
    y_true=np.squeeze(y_true)
    y_pred=np.squeeze(y_pred)
    if y_true.shape != y_pred.shape:
        raise('Shapes for Dice not equaL!')

    t = np.ones(y_pred.shape) * thresh
    y_pred = np.greater_equal(y_pred, t) #threshold on value
    y_true = np.greater_equal(y_true, t)
    inter = np.logical_and(y_pred, y_true) #find pixels that intersect
    interpix = np.count_nonzero(inter) # count the pixels in inter
    y_predpix = np.count_nonzero(y_pred) #count the pixels in pred
    y_truepix = np.count_nonzero(y_true) # count the pixels in true
    dice = 2*interpix/(y_predpix+y_truepix+1e-5) # calculate Dice
    return dice




def MOACC(image_path, rtstruct_path):#, gtstruct_path):
    organlist = ['Duodenum','Stomach','Colon','Bowel_Small','Liver','Kidney_R','Kidney_L']
    colorlist = [[255,0,255],[0,178,47],[255,255,0],[0,255,255],[160,32,240], [128,0,128], [0,128,0], [0,128,128]]
    #struct_path = str(rtstruct_path)
    struct_file = os.listdir(rtstruct_path)[0]

    MRvolume, contourvolume, pixelW = contourdata.get_DL_data_multiorgan(rtstruct_path,image_path,organ_list=organlist)
    #dicom_images2, dicom_gt_masks, pixelW2 = contourdata.get_DL_data_multiorgan(gtstruct_path,image_path,organ_list=organlist)

    dl_masks_ACC = denseunet.ApplyDenseUNetACC_multiorgan(MRvolume,contourvolume)
    
    mat_out = {'MR':MRvolume,'Init':contourvolume,'ACC':dl_masks_ACC,'pixel':pixelW}
    sio.savemat(os.path.join(outdir,patient+'_'+date+'.mat'),mat_out)

    ## Open RT structure File
    rtstruct = RTStructBuilder.create_new(dicom_series_path=image_path)
    #remove_all_ROI(rtstruct)

    shapes = dl_masks_ACC.shape
    if shapes[2]<100:
        extraslice = np.zeros([shapes[0],shapes[1],1])
        dl_masks_ACC = np.concatenate((dl_masks_ACC,extraslice),axis=2)

    for o, organID in enumerate(organlist):
        organmask = dl_masks_ACC == o+1

        rtstruct.add_roi(mask=organmask, name=organID, approximate_contours = False)


    ## After finishing with all organs, we save the RT struct with the new contours.
    ## Will save RTstruct file as [PATIENT]_[DATE].dcm. May want to overwrite file
    ## that's already there instead.

    rtstruct.save(os.path.join(outdir,patient+'_'+date+'.dcm'))

    return None


times = []
toppath = '/mnt/G/Physicist/people/Sarosiek/1_DenseUNet_DelRec/multiorgan_data/ApplicationTests/Manuscript2025/' #Path to DICOM Data
outpath = '/mnt/G/Physicist/people/Sarosiek/1_DenseUNet_DelRec/multiorgan_data/ApplicationTests/Manuscript2025/MOACC-Clinical/' #Path to save new structures
patients = os.listdir(toppath+'MRL')
for patient in patients:

    os.mkdir(os.path.join(outpath,patient))
    
    dates = os.listdir(os.path.join(toppath,'MRL',patient))
    for date in dates:
        image_path = os.path.join(toppath,'MRL',patient,date)
        #gtstruct_path = os.path.join(toppath,'GT',patient)
        rtstruct_path = os.path.join(toppath,'Clinical',patient,date)
        
        os.mkdir(os.path.join(outpath,patient, date))
        outdir = os.path.join(outpath, patient, date)
    
        start = time.time()
    
        MOACC(image_path, rtstruct_path)#,gtstruct_path)
    
        end = time.time()
        times.append(end-start)

print('times',times)
