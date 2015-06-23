# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 12:28:03 2015
@author: subru
"""

import mha
import numpy as np
import os
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
path = 'data/Normalized_Training/'
folder = 'brats_tcia_pat105_1/'

Flair = []
T1 = []
T2 = []
T_1c = []
Truth = []
Folder = []

for subdir, dirs, files in os.walk('data/Normalized_Training'):
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
    print 'Iteration : ',image_iterator+1
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
        x_stop = np.max(x_span) + padding
        y_start = np.min(y_span) - padding
        y_stop = np.max(y_span) + padding
        
        iterate_x = x_start
        while iterate_x <= x_stop:    
            iterate_y = y_start
            while iterate_y <= y_stop:
                temp_patch = np.zeros(patch_pixels*4)
                if iterate_x < patch_size/2:
                    print 'Correction X min'
                    iterate_x = (patch_size/2) +1
                elif iterate_x > x_dim - (patch_size/2):
                    print 'Correction X max'
                    iterate_x = x_dim - (patch_size/2) -1
                if iterate_y < (patch_size/2):
                    print 'Correction Y min'
                    iterate_y = (patch_size/2)
                elif iterate_y > y_dim - (patch_size/2):
                    print 'Correction Y max'
                    iterate_y = y_dim - (patch_size/2) - 1
                    
                #print (iterate_x,iterate_y)
                Flair_patch = Flair_slice[(iterate_x-(patch_size/2)):(iterate_x+(patch_size/2)+1), (iterate_y-(patch_size/2)):(iterate_y+(patch_size/2)+1)]
                temp_patch[0:patch_pixels] = np.asarray(Flair_patch).reshape(-1)
                
                if np.sum((temp_patch[0:patch_pixels]!=0).astype(int))<threshold:
                    iterate_y = iterate_y+pixel_offset
                    count1 = count1+1
                    continue
                
                T1_patch = T1_slice[(iterate_x-(patch_size/2)):(iterate_x+(patch_size/2)+1), (iterate_y-(patch_size/2)):(iterate_y+(patch_size/2)+1)]
                temp_patch[patch_pixels:2*patch_pixels] = np.asarray(T1_patch).reshape(-1)
                
                T2_patch = T2_slice[(iterate_x-(patch_size/2)):(iterate_x+(patch_size/2)+1), (iterate_y-(patch_size/2)):(iterate_y+(patch_size/2)+1)]
                temp_patch[2*patch_pixels:3*patch_pixels] = np.asarray(T2_patch).reshape(-1)
                
                T_1c_patch = T_1c_slice[(iterate_x-(patch_size/2)):(iterate_x+(patch_size/2)+1), (iterate_y-(patch_size/2)):(iterate_y+(patch_size/2)+1)]
                temp_patch[3*patch_pixels:4*patch_pixels] = np.asarray(T_1c_patch).reshape(-1)
                
                #truth_patch = Truth_slice[(iterate_x-(patch_size/2)):(iterate_x+(patch_size/2)+1), (iterate_y-(patch_size/2)):(iterate_y+(patch_size/2)+1)]
                #truth_patch = np.asarray(truth_patch).reshape(-1)
                #if np.sum((truth_patch!=0).astype(int))<threshold:
                #    iterate_y = iterate_y+pixel_offset
                #    count5 = count5+1
                #    continue
                
                pixel_truth = np.asarray(Truth_slice[iterate_x,iterate_y])
                patches = np.vstack([patches,temp_patch])
                ground_truth = np.vstack([ground_truth, pixel_truth])
                iterate_y = iterate_y + pixel_offset
            iterate_x = iterate_x + pixel_offset
            
        #plt.imshow(Flair_slice[x_start:x_stop,y_start:y_stop])
    #print count1

print 'Number of rows, columns in patches array : ', np.size(patches,axis=0), np.size(patches,axis=1)


print 'Number of non-zeros in ground truth : ', np.sum((ground_truth!=0).astype(int))
print 'Number of zeros in ground truth : ', np.sum((ground_truth==0).astype(int))

print
print 'No. of 1 : ', np.sum((ground_truth==1).astype(int))
print 'No. of 2 : ', np.sum((ground_truth==2).astype(int))
print 'No. of 3 : ', np.sum((ground_truth==3).astype(int))
print 'No. of 4 : ', np.sum((ground_truth==4).astype(int))

#np.save('Training_patches.npy',patches)
#np.save('Training_labesl.npy',ground_truth)