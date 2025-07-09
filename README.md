# Brain Tumors
# introduction
This repository hosts the **Brain Tumors dataset from Kaggle, consisting of MRI scans of brain tumors. It’s ideal for training and evaluating machine-learning models for classification, segmentation, or detection tasks.
# Tools Used
Python 3.8+

NumPy – numerical operations

Pandas – data handling

OpenCV – image I/O & preprocessing

Matplotlib – plotting & visualization

TensorFlow / Keras – deep-learning framework (optional)

scikit-learn – ML utilities

# Libraries
 Python script that imports all the core libraries for building and training a convolutional neural network (CNN) with TensorFlow/Keras. Specifically, it shows:
  ![code-snapshot](https://github.com/user-attachments/assets/302c9c89-a8c3-47f2-b941-d1bd439c3f92)

### Standard utilities
- `import os` — OS interface (files, directories, paths, env)  
- `import time` — timing utilities (timestamps, delays)

<img src="https://github.com/user-attachments/assets/6d5b850c-fc6b-46cb-860a-c0278cdebc77" alt="import os & time" width="450" />

### Visualization
- `import matplotlib.pyplot as plt` — 2D plotting library for visualizing data and images

<img src="https://github.com/user-attachments/assets/60f492fd-9ae6-4f17-8234-66285edf1617" alt="matplotlib import" width="450" />

### Model definition
- `Sequential` — simple linear stack  
- `Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input` — conv, pool, global pool, FC, dropout, input

<img src="https://github.com/user-attachments/assets/f5772848-869a-4238-aead-63fcdacc6e9e" alt="model definition imports" width="450" />

### Optimization
- `Adam` — combines momentum and per-parameter adaptive learning rates for fast convergence

<img src="https://github.com/user-attachments/assets/3dad17f3-a89a-4bc4-a00e-28bde8754d33" alt="Adam optimizer import" width="450" />

### Data handling
- `ImageDataGenerator` — real-time image augmentation

<img src="https://github.com/user-attachments/assets/2db6421b-85a5-40b5-b4d8-56a7d5a8dfd4" alt="ImageDataGenerator import" width="450" />

### Training callbacks
- `ModelCheckpoint` — auto-save best model

<img src="https://github.com/user-attachments/assets/c6880660-5692-4ced-b899-d8982fd15c4d" alt="ModelCheckpoint import" width="450" />

## Data Preparation
<img src="https://github.com/user-attachments/assets/d071708b-f938-4c91-907f-4aaa40f004d2" alt="ImageDataGenerator import" width="450" />

## Training and validation accuracy chart
<img src="https://github.com/user-attachments/assets/979d46cd-311b-4724-aa42-22dac39b0660" alt="ImageDataGenerator import" width="450" />
<img src="https://github.com/user-attachments/assets/965af988-60f1-48cd-8bbd-4e29184cae2d" alt="ImageDataGenerator import" width="450" />

## Training and verification loss drawing
<img src="https://github.com/user-attachments/assets/29987744-cce8-4318-a3ac-8bc63e01c6f1" alt="ImageDataGenerator import" width="450" />
<img src="https://github.com/user-attachments/assets/04412bcf-8fb8-418d-b607-365ddfbf4a0a" alt="ImageDataGenerator import" width="450" />

## Training summary 
<img src="https://github.com/user-attachments/assets/ad22d690-99e1-4127-99d2-2d9b991ce4a1" alt="ImageDataGenerator import" width="450" />

## Normal_tumor 
<img width="876" alt="Screenshot 1447-01-14 at 5 59 47 AM" src="https://github.com/user-attachments/assets/2801dca4-3824-4482-910c-f8caff72763b" />

## Glioma_tumor
<img width="878" alt="Screenshot 1447-01-14 at 5 48 35 AM" src="https://github.com/user-attachments/assets/ef7012bb-fc3f-4f11-9b54-1e3aadc72a1e" />

## Meningioma_tumor
<img width="878" alt="Screenshot 1447-01-14 at 5 55 57 AM" src="https://github.com/user-attachments/assets/f73dd397-21a9-49a5-a88c-dd4fc37c57d7" />

## Pituitary_tumor
<img width="727" alt="Screenshot 1447-01-14 at 6 03 32 AM" src="https://github.com/user-attachments/assets/707941d6-850e-4724-a4d5-bafcf6879468" />
