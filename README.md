Deep Learning-Refined Contour Propagation

Preprocessing Code: PrepareData_training_multiorgan_V2.py
Reads the DICOM Image and RT Structure files and preprocesses them for DL-RCP training. Saves each slice as a .npy file. 

Model Training Code: DenseUNet-v3_multiorgan.py
Dependencies: 
  LoadData_v4_python_aug.py
  DataGenerator_v4_multiorgan.py
  DenseLayers.py
Model architecture inspired by a 2D Dense UNet model. 

Production Use: MOACCWorkflow_final.py
Dependencies:
  contourdata.py
  denseunet.py
Reads the DICOM images and RTStructure Set, applies the trained model, and produces a second RT Structure Set with the updated contours.
