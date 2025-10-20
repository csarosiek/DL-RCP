# Deep Learning-Refined Contour Propagation

The DL-RCP tool uses a deep learning model to improve contour quality. It is inspired by a 2D Dense UNet model taking an MR slice and inaccurate contours and provides updated contours. The tool is designed to improve 7 abdominal contours (stomach, duodenum, small bowel, colon, liver, left kidney, right kidney) on MRI.
This repository holds the code required to 1) preprocess training data, 2) train a model, and 3) use the DL-RCP tool in production. 

## Python Dependencies:
os

numpy

rt_utils

scipy

time

pydicom

cv2

matplotlib

tensorflow

multiprocessing

pandas


## Preprocessing Code: 
PrepareData_training_multiorgan_V2.py

### Use
Reads the DICOM Image and RT Structure files and preprocesses them for DL-RCP training. Saves each slice as a .npy file. 

To use, change the GT_path and Init_path variables to the paths with the ground truth RTSS and the inaccurate initial RTSS, respectively. Within each of these paths, there should be identical list of directories with the patient labels. The code will match the patients between each directory. 

Each slice is saved as a 4D numpy array with [0,x,y,0] = MR slice, [0,x,y,1] = inaccurate contour mask, [0,x,y,2] = GT contour mask. These can be combined later into a single 4D array for easy loading into model training code. 

You can update the organ names in the organlist variable. This is currently hard-coded in each piece of code and requires manual updates in each piece of code if changed. 

Uses multiprocessing to process multiple DICOM images at a time and speed up processing.

Also included are lines to plot the data throughout the preprocessing method that can be useful for troubleshooting. These are currently commented out. 

## Model Training Code: 
DenseUNet-v3_multiorgan.py

Dependencies: LoadData_v4_python_aug.py, DataGenerator_v4_multiorgan.py, DenseLayers.py
  
Model architecture inspired by a 2D Dense UNet model. 

To train, update the hyperparameters in Lines 65-74 and the directories in Lines 77-80. 

## Production Use: 
MOACCWorkflow_final.py

Dependencies: contourdata.py, denseunet.py
  
Reads the DICOM images and RTStructure Set, applies the trained model, and produces a second RT Structure Set with the updated contours.

To run, change the directories in Lines 127-128. The code will look for the following model './multiorgan_data/organs_7/model_optimal/ACC_multiorgan.h5'
