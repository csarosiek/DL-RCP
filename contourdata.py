# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 12:42:23 2023

This file contains all the functions necessary to read the DICOM image
and rt structure files.

@author: csarosiek
"""

import os
import pydicom
#from pydicom import dcmread
#import readdcm
from tools import data
import numpy as np
import cv2
from matplotlib.path import Path
import matplotlib.pyplot as plt

def get_DL_data(rtstruct_path=None,image_path=None, organ_name=''):
    print(organ_name)
    # find DL contour file
    dl_contour_file=os.listdir(rtstruct_path)[-1]#get_contour_file(rtstruct_path)
    dl_contour_path=os.path.join(rtstruct_path, dl_contour_file)

    dicom = data.DicomData()
    dicom_slices, thickness, offset, size = read_dicom_files(image_path)#, fullname=True)
    dicom_images = reconstruct_3d_volume(dicom_slices)
    print('Got images')
    contourpts, contourpositions = findContours(dl_contour_path,organ_name)
    if contourpts == []:
        dicom_dl_masks = []
    else:
        dicom_dl_masks = getDLmasks(contourpts, contourpositions, position=list(dicom_slices.keys()), size=size, pixelsize=thickness,offset=offset)
    # dicom_dl_masks = get_organ_mask_CS(organ=organ_name, struct_file=dl_contour_path,
    #                                 size=tuple(dicom_images.shape),orig=dicom.get_origin(),
    #                                 thickness=thickness,position=position,location=list(sorted(dicom_slices.keys())))
    #ds=dcmread(dicom_files[5],force=True)
    #pixelW=ds.PixelSpacing[0]
    print('Got masks')
    return dicom_images, dicom_dl_masks, thickness


def get_DL_data_multiorgan(rtstruct_path=None,image_path=None, organ_list=[]):
    #print(organ_name)
    # find DL contour file
    dl_contour_file=os.listdir(rtstruct_path)[-1]#get_contour_file(rtstruct_path)
    dl_contour_path=os.path.join(rtstruct_path, dl_contour_file)

    dicom = data.DicomData()
    dicom_slices, thickness, offset, size = read_dicom_files(image_path)#, fullname=True)
    dicom_images = reconstruct_3d_volume(dicom_slices)
    print('Got images')
    shape = np.shape(dicom_images)
    dicom_dl_allmasks = np.zeros(np.shape(dicom_images))
    for o, organ_name in enumerate(organ_list):
        print(organ_name)
        contourpts, contourpositions = findContours(dl_contour_path,organ_name)
        if contourpts == []:
            dicom_dl_masks = []
        else:
            dicom_dl_masks = getDLmasks(contourpts, contourpositions, position=list(dicom_slices.keys()), size=size, pixelsize=thickness,offset=offset)
            dicom_dl_allmasks[(dicom_dl_masks == True) & (dicom_dl_allmasks == 0)] = o+1


    # dicom_dl_masks = get_organ_mask_CS(organ=organ_name, struct_file=dl_contour_path,
    #                                 size=tuple(dicom_images.shape),orig=dicom.get_origin(),
    #                                 thickness=thickness,position=position,location=list(sorted(dicom_slices.keys())))
    #ds=dcmread(dicom_files[5],force=True)
    #pixelW=ds.PixelSpacing[0]
    print('Got masks')
    return dicom_images, dicom_dl_allmasks, thickness



def get_contour_file(path):
    """
    Get contour file from a given path by searching for ROIContourSequence
    inside dicom data structure.
    More information on ROIContourSequence available here:
    http://dicom.nema.org/medical/dicom/2016c/output/chtml/part03/sect_C.8.8.6.html
    Inputs:
            path (str): path of the the directory that has DICOM files in it, e.g. folder of a single patient
    Return:
        contour_file (str): name of the file with the contour
    """
    # handle `/` missing
    #if path[-1] != '/':
    #    path += '/'
    # get .dcm contour file
    fpaths = [path + f for f in os.listdir(path) if '.dcm' in f]
    n = 0
    contour_file = None
    for fpath in fpaths:
        f = pydicom.read_file(fpath,force=True)
        if 'ROIContourSequence' in dir(f):
            contour_file = fpath.split('/')[-1]
            n += 1
    if n > 1:
        print("There are multiple contour files, returning the last one!")
    if contour_file is None:
        print("No contour file found in directory")

    return contour_file



def read_dicom_files(folder_path):
    dicom_slices = {}

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path)[:-1]:
        file_path = os.path.join(folder_path, filename)

        # Check if the file is a DICOM file based on content
        try:
            ds = pydicom.dcmread(file_path)
        except pydicom.errors.InvalidDicomError:
            print('Invalid DICOM Error')
            continue  # Skip non-DICOM files
        #print(ds.Modality)
        # Extract relevant information (e.g., slice location)
        pixel_size  = ds.PixelSpacing
        slicethick = ds.SliceThickness
        slice_location = ds.SliceLocation  # Modify this based on your DICOM header
        thickness = [pixel_size[0],pixel_size[1],slicethick]
        position = ds.ImagePositionPatient
        rows = ds.Rows
        columns = ds.Columns

        # Group DICOM files based on slice location
        if slice_location in dicom_slices:
            dicom_slices[slice_location].append(ds)
        else:
            dicom_slices[slice_location] = [ds]



    dicom_slices = dict(sorted(dicom_slices.items()))
    size = [rows, columns, len(dicom_slices)]

    return dicom_slices, thickness, position, size

def reconstruct_3d_volume(dicom_slices):
    # Sort slices based on slice location
    sorted_slices = sorted(dicom_slices.items())

    # Extract pixel data from each slice
    slices_pixel_data = [np.array(ds_list[0].pixel_array) for (_, ds_list) in sorted_slices]

    # Create a 3D volume
    volume = np.stack(slices_pixel_data, axis=-1)

    return volume


def findContours(rtstruct,organ):
    slicepoints = []
    slicezpos = []
    refnumber = 999
    dcmfile = pydicom.dcmread(rtstruct)
    if dcmfile.Modality != 'RTSTRUCT':
        print('Not an RTStruct File')
    else:
        i = 0
        while i < len(dcmfile.StructureSetROISequence):
            if organ in dcmfile.StructureSetROISequence[i].ROIName:
                refnumber = dcmfile.StructureSetROISequence[i].ROINumber
            i += 1
        if refnumber == 999:
            slicepoints = []
            slicezpos = []
        else:
            for oar in dcmfile.ROIContourSequence:
                if oar.ReferencedROINumber == refnumber:
                    ptslist = np.asarray(oar.ContourSequence[0].ContourData)
                    pts = np.ndarray.reshape(ptslist,int(len(ptslist)/3),3)
                    slicezpos.append(pts[0,2])
                    slicepoints.append(pts[:,:2])
                    j = 1
                    while j < len(oar.ContourSequence):
                        ptslist = np.asarray(oar.ContourSequence[j].ContourData)
                        pts = np.ndarray.reshape(ptslist,int(len(ptslist)/3),3)
                        slicepoints.append(pts[:,:2])
                        slicezpos.append(pts[0,2])
                        j += 1
    return slicepoints, slicezpos

#converts the contour points from mm to pixel coordinates
def convert2pixel(pixelsize,offset,contours):
    xmm = contours[:,0]
    ymm = contours[:,1]
    xpixel = (xmm - offset[0])/pixelsize[0]
    ypixel = (ymm - offset[1])/pixelsize[1]
    sliceptspixel = []
    for i in range(0,len(xmm)):
        sliceptspixel.append([xpixel[i],ypixel[i]])
    return np.asarray(sliceptspixel)


#Creates a binary mask from the contours
def createMask(dimensions, contours):
    x, y = np.meshgrid(np.arange(dimensions[1]),np.arange(dimensions[0]))
    x, y = x.flatten(), y.flatten()

    points = np.vstack((x,y)).T

    path = Path(contours)
    grid = path.contains_points(points)
    grid = grid.reshape((dimensions[0],dimensions[1]))

    return grid

def getDLmasks(slicepoints, slicezpos, position, size, pixelsize, offset):
    dl_masks = np.zeros(size)
    slicesize = size[:2]
    for j,z in enumerate(position):
        im_temp = np.empty(slicesize)
        ## DL contours
        check = 0
        for i,z_DL in enumerate(slicezpos):
            if abs(z_DL-z) < 0.75:
                check += 1
                contourpix_DL = convert2pixel(pixelsize, offset, slicepoints[i])
                contourmask_DL = createMask(slicesize, contourpix_DL)
                if i == 0:
                   slicemask_DL = contourmask_DL
                else:
                    if z_DL == slicezpos[i-1]:
                        slicemask_DL = np.ma.mask_or(contourmask_DL, slicemask_DL)
                    else:
                        slicemask_DL = contourmask_DL
            if check == 0:
                slicemask_DL = np.zeros(slicesize)
            im_temp = slicemask_DL
        dl_masks[:,:,j] = im_temp
    return dl_masks.astype(bool)

# def get_organ_mask_CS(organ='', struct_file=None, size=(512, 512, 10), orig=[0.,471.07,-195.],
#                    thickness=[0.9219,0.9219,3.0],position=[0,0,0],location=[0,1,2]):
#     """!
#     @brief Getting ROI structure from RTSTRUCT file

#     """
#     print(organ)
#     if organ == '':
#         return
#     # read RTSTRUCT file
#     struct_data = pydicom.dcmread(struct_file)
#     # get organ id
#     organ_id = None
#     organ_dict = struct_data.StructureSetROISequence
#     i = 0
#     while i < len(organ_dict):
#         if organ_dict[i].ROIName == organ:
#             organ_id = organ_dict[i].ROINumber
#             break
#         i += 1
#     if organ_id == None:
#         return
#     # get pixel thicknesses
#     #xsize, ysize, zsize = self.get_pixel_thickness()
#     mask = np.zeros(size)
#     # loop over contour sequence
#     for ROI in struct_data.ROIContourSequence:
#         #self._logger.debug(ROI.ReferencedROINumber)
#         if ROI.ReferencedROINumber == organ_id:
#             contour_sequence = ROI.ContourSequence
#             #print(contour_sequence)
#             break
#     #print(contour_sequence)
#     return create_mask(contour_sequence,size=size, orig=orig,
#                    thickness=thickness,position=position, location=location)

# def get_slice_directions(orientation):
#     #orientation = series_slice.ImageOrientationPatient
#     row_direction = np.array(orientation[:3])
#     column_direction = np.array(orientation[3:])
#     slice_direction = np.cross(row_direction, column_direction)
#     return row_direction, column_direction, slice_direction

# def create_mask(contour_sequence,size=(512, 512, 10), orig=[0.,471.07,-195.],
#                    thickness=[0.9219,0.9219,3.0],position=[0,0,0],location=[0,1,2]):
#     # mask
#     mask = np.zeros(size).astype(bool)
#     # transformation matrix
#     # offset = np.array(position)
#     # #print(thickness)
#     # row_spacing = thickness[0]
#     # column_spacing = thickness[1]
#     # slice_spacing = thickness[2]
#     # row_direction, column_direction, slice_direction = get_slice_directions([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
#     # linear = np.identity(3, dtype=np.float32)
#     # linear[0,:3] = row_direction / row_spacing
#     # linear[1,:3] = column_direction / column_spacing
#     # linear[2,:3] = slice_direction / slice_spacing
#     # transformation_matrix = np.identity(4, dtype=np.float32)
#     # transformation_matrix[:3,:3] = linear
#     # transformation_matrix[:3,3] = offset.dot(-linear.T)
#     # loop through each slice of the series
#     for i, z in enumerate(location):
#         slice_contour_data = get_slice_contour_data(z, contour_sequence)
#         if len(slice_contour_data):
#             slice_mask = mask[:,:,i]
#             for contour_coords in slice_contour_data:
#                 fill_mask = get_contour_fill_mask(slice_mask, contour_coords, transformation_matrix)
#                 slice_mask[fill_mask == 1] = np.invert(slice_mask[fill_mask == 1])
#             mask[:, :, i] = slice_mask
#     return mask

# def get_contour_fill_mask(slice_mask, contour_coords, transformation_matrix):
#     reshaped_contour_data = np.reshape(contour_coords, [len(contour_coords) // 3, 3])
#     translated_contour_data = apply_transformation_to_3d_points(reshaped_contour_data, transformation_matrix)
#     polygon = [np.around([translated_contour_data[:,:2]]).astype(np.int32)]
#     fill_mask = slice_mask.astype(np.uint8)
#     cv2.fillPoly(img=fill_mask, pts=polygon, color=1)
#     return fill_mask

# def apply_transformation_to_3d_points(points, transformation_matrix):
#     vec = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
#     return vec.dot(transformation_matrix.T)[:,:3]

# def get_slice_contour_data(z, contour_sequence):
#     slice_contour_data = []
#     for contour in contour_sequence:
#         for contour_image in contour.ContourImageSequence:
#             if abs(z - contour.ContourData[2]) < 0.5:
#                 slice_contour_data.append(contour.ContourData)
#     return slice_contour_data
