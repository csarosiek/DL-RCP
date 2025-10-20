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

Reads the DICOM Image and RT Structure files and preprocesses them for DL-RCP training. Saves each slice as a .npy file. 

## Model Training Code: 
DenseUNet-v3_multiorgan.py

Dependencies: 

  LoadData_v4_python_aug.py
  
  DataGenerator_v4_multiorgan.py
  
  DenseLayers.py
  
Model architecture inspired by a 2D Dense UNet model. 

## Production Use: 
MOACCWorkflow_final.py

Dependencies:

  contourdata.py
  
  denseunet.py
  
Reads the DICOM images and RTStructure Set, applies the trained model, and produces a second RT Structure Set with the updated contours.
