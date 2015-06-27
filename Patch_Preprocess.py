# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:28:03 2015
@author: subru
"""

import mha
import numpy as np
import os
from sklearn.feature_extraction import image
#from matplotlib import pyplot as plt

#Initialize user variables
patch_size = 11
patch_pixels = patch_size*patch_size
pixel_offset = int(patch_size*0.7)
padding = patch_size/2
threshold = patch_pixels*0.3
patches = np.zeros(patch_pixels*4)
ground_truth = np.zeros(1)

#paths to images
path = '../BRATS/10_1/training/'

Flair = []
T1 = []
T2 = []
T_1c = []
Truth = []
Folder = []

for subdir, dirs, files in os.walk(path):
    if len(Flair) == 20:
        break
    for file1 in files:     
            
        #print file1
        if file1[-3:]=='mha' and 'Flair' in file1:
            Flair.append(file1)
            Folder.append(subdir+'/')
        elif file1[-3:]=='mha' and 'T1' in file1:
            T1.append(file1)
        elif file1[-3:]=='mha' and 'T2' in file1:
            T2.append(file1)
        elif file1[-3:]=='mha' and 'T_1c' in file1:
            T_1c.append(file1)
        elif file1[-3:]=='mha' and 'OT' in file1:
            Truth.append(file1)
            
number_of_images = len(Flair)
print 'Number of images : ', number_of_images


count1, count2, count3, count4, count5 = 0,0,0,0,0
for image_iterator in range(number_of_images):
    print 'Image number : ',image_iterator+1
    print 'Folder : ', Folder[image_iterator]
    Flair_image = mha.new(Folder[image_iterator]+Flair[image_iterator])
    T1_image = mha.new(Folder[image_iterator]+T1[image_iterator])
    T2_image = mha.new(Folder[image_iterator]+T2[image_iterator])
    T_1c_image = mha.new(Folder[image_iterator]+T_1c[image_iterator])
    Truth_image = mha.new(Folder[image_iterator]+Truth[image_iterator])
    
    Flair_image = Flair_image.data
    T1_image = T1_image.data
    T2_image = T2_image.data
    T_1c_image = T_1c_image.data
    Truth_image = Truth_image.data
    
    x_span,y_span,z_span = np.where(Truth_image!=0)
    
    start_slice = min(z_span)
    stop_slice = max(z_span)
    
    for i in range(start_slice, stop_slice+1):    
        Flair_slice = np.transpose(Flair_image[:,:,i])
        T1_slice = np.transpose(T1_image[:,:,i])
        T2_slice = np.transpose(T2_image[:,:,i])
        T_1c_slice = np.transpose(T_1c_image[:,:,i])
        Truth_slice = np.transpose(Truth_image[:,:,i])
        
        x_dim,y_dim = np.size(Flair_slice,axis=0), np.size(Flair_slice, axis=1)
        
        x_span,y_span = np.where(Truth_slice!=0)
        if len(x_span)==0 or len(y_span)==0:
            continue
        x_start = np.min(x_span) - padding
        x_stop = np.max(x_span) + padding+1
        y_start = np.min(y_span) - padding
        y_stop = np.max(y_span) + padding+1
        
        Flair_patch = image.extract_patches(Flair_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
        T1_patch = image.extract_patches(T1_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
        T2_patch = image.extract_patches(T2_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
        T_1c_patch = image.extract_patches(T_1c_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
        Truth_patch = image.extract_patches(Truth_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
        
        #print '1. truth dimension :', Truth_patch.shape
        
        Flair_patch = Flair_patch.reshape(Flair_patch.shape[0]*Flair_patch.shape[1], patch_size*patch_size)
        T1_patch = T1_patch.reshape(T1_patch.shape[0]*T1_patch.shape[1], patch_size*patch_size)
        T2_patch = T2_patch.reshape(T2_patch.shape[0]*T2_patch.shape[1], patch_size*patch_size)  
        T_1c_patch = T_1c_patch.reshape(T_1c_patch.shape[0]*T_1c_patch.shape[1], patch_size*patch_size)
        Truth_patch = Truth_patch.reshape(Truth_patch.shape[0]*Truth_patch.shape[1], patch_size, patch_size)
        
        #print '2. truth dimension :', Truth_patch.shape
        
        slice_patch = np.concatenate([Flair_patch, T1_patch, T2_patch, T_1c_patch], axis=1)
        Truth_patch = Truth_patch[:,(patch_size-1)/2,(patch_size-1)/2]
        Truth_patch = np.array(Truth_patch)
        Truth_patch = Truth_patch.reshape(len(Truth_patch),1)
        #print '3. truth dimension :', Truth_patch.shape
        
        patches = np.vstack([patches,slice_patch])
        ground_truth = np.vstack([ground_truth, Truth_patch])


print 'Number of non-zeros in ground truth : ', np.sum((ground_truth!=0).astype(int))
print 'Number of zeros in ground truth : ', np.sum((ground_truth==0).astype(int))

print 'No. of 1 : ', np.sum((ground_truth==1).astype(int))
print 'No. of 2 : ', np.sum((ground_truth==2).astype(int))
print 'No. of 3 : ', np.sum((ground_truth==3).astype(int))
print 'No. of 4 : ', np.sum((ground_truth==4).astype(int))

ground_truth = ground_truth.reshape(len(ground_truth))
np.save('Training_patches.npy',patches)
np.save('Training_labels.npy',ground_truth)
#print ground_truth.shape
#print patches.shape
#np.save('Testing_patches.npy',patches)
#np.save('Testing_labesl.npy',ground_truth)