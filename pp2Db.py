# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 17:13:17 2015

@author: bmi
"""

import mha
import numpy as np
import os
from sklearn.feature_extraction import image
from random import shuffle
#from matplotlib import pyplot as plt

def B_Patch_Preprocess_recon_2D(patch_size_x=5,patch_size_y=5,prefix='Sda',in_root='',out_root='',recon_flag=True):
    
    #Initialize user variables
    patch_size = patch_size_x
    patch_pixels = patch_size*patch_size
    pixel_offset = int(patch_size*0.7)
    padding = patch_size*2
    threshold = patch_pixels*0.3
    if recon_flag == False:
        recon_num = 4
    if recon_flag == True:
        recon_num = 5
    patches = np.zeros(patch_pixels*recon_num)
    ground_truth = np.zeros(1)
    
    #paths to images
    path = in_root
    
    Flair = []
    T1 = []
    T2 = []
    T_1c = []
    Truth = []
    Recon=[]
    Folder = []
    
    for subdir, dirs, files in os.walk(path):
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
            #elif file1[-3:]=='mha' and 'Recon' in file1:
            #    Recon.append(file1)
                
    number_of_images = len(Flair)
    print 'Number of images : ', number_of_images
    
    
    for image_iterator in range(number_of_images):
        print 'Iteration : ',image_iterator+1
        print 'Folder : ', Folder[image_iterator]
        Flair_image = mha.new(Folder[image_iterator]+Flair[image_iterator])
        T1_image = mha.new(Folder[image_iterator]+T1[image_iterator])
        T2_image = mha.new(Folder[image_iterator]+T2[image_iterator])
        T_1c_image = mha.new(Folder[image_iterator]+T_1c[image_iterator])
        if recon_flag == True:
            Recon_image = mha.new(Folder[image_iterator]+Recon[image_iterator])
        Truth_image = mha.new(Folder[image_iterator]+Truth[image_iterator])
        
        Flair_image = Flair_image.data
        T1_image = T1_image.data
        T2_image = T2_image.data
        T_1c_image = T_1c_image.data
        if recon_flag == True:
            Recon_image=Recon_image.data
        Truth_image = Truth_image.data
        
        x_span,y_span,z_span = np.where(Truth_image!=0)
        
        start_slice = min(z_span)
        stop_slice = max(z_span)
        image_patch = np.zeros(patch_size*patch_size*recon_num)
        image_label = np.zeros(1)
        for i in range(start_slice, stop_slice+1):    
            Flair_slice = np.transpose(Flair_image[:,:,i])
            T1_slice = np.transpose(T1_image[:,:,i])
            
            T2_slice = np.transpose(T2_image[:,:,i])
            T_1c_slice = np.transpose(T_1c_image[:,:,i])
            if recon_flag==True:
                Recon_slice = np.transpose(Recon_image[:,:,i])      
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
            if recon_flag==True:
                Recon_patch = image.extract_patches(Recon_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)      
            Truth_patch = image.extract_patches(Truth_slice[x_start:x_stop, y_start:y_stop], patch_size, extraction_step = pixel_offset)
            
            #print '1. truth dimension :', Truth_patch.shape
            
            Flair_patch = Flair_patch.reshape(Flair_patch.shape[0]*Flair_patch.shape[1], patch_size*patch_size)
            T1_patch = T1_patch.reshape(T1_patch.shape[0]*T1_patch.shape[1], patch_size*patch_size)
            T2_patch = T2_patch.reshape(T2_patch.shape[0]*T2_patch.shape[1], patch_size*patch_size)  
            T_1c_patch = T_1c_patch.reshape(T_1c_patch.shape[0]*T_1c_patch.shape[1], patch_size*patch_size)
            if recon_flag==True:
                Recon_patch = Recon_patch.reshape(Recon_patch.shape[0]*Recon_patch.shape[1], patch_size*patch_size)        
            Truth_patch = Truth_patch.reshape(Truth_patch.shape[0]*Truth_patch.shape[1], patch_size, patch_size)
            
            #print '2. truth dimension :', Truth_patch.shape
            if recon_flag == True:
                slice_patch = np.concatenate([Flair_patch, T1_patch, T2_patch, T_1c_patch,Recon_patch], axis=1)
            else:
                slice_patch = np.concatenate([Flair_patch, T1_patch, T2_patch, T_1c_patch], axis=1)
            Truth_patch = Truth_patch[:,(patch_size-1)/2,(patch_size-1)/2]
            Truth_patch = np.array(Truth_patch)
            Truth_patch = Truth_patch.reshape(len(Truth_patch),1)
            #print '3. truth dimension :', Truth_patch.shape
            
            image_patch = np.vstack([image_patch,slice_patch])
            image_label = np.vstack([image_label, Truth_patch])
        num_of_class = []
        for i in xrange(1,5):
            num_of_class.append(np.sum((image_label==i).astype(int)))
        max_num = max(num_of_class)
        max_num_2 = max(x for x in num_of_class if x!=max_num)
        
        Flair_patch = image.extract_patches(Flair_image[:,:,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        Flair_patch = Flair_patch.reshape(Flair_patch.shape[0]*Flair_patch.shape[1]*Flair_patch.shape[2], patch_size_x*patch_size_y)
        T1_patch = image.extract_patches(T1_image[:,:,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        T1_patch = T1_patch.reshape(T1_patch.shape[0]*T1_patch.shape[1]*T1_patch.shape[2], patch_size_x*patch_size_y)
        T2_patch = image.extract_patches(T2_image[:,:,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        T2_patch = T2_patch.reshape(T2_patch.shape[0]*T2_patch.shape[1]*T2_patch.shape[2], patch_size_x*patch_size_y)
        T_1c_patch = image.extract_patches(T_1c_image[:,:,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        T_1c_patch = T_1c_patch.reshape(T_1c_patch.shape[0]*T_1c_patch.shape[1]*T_1c_patch.shape[2], patch_size_x*patch_size_y)
        Truth_patch = image.extract_patches(Truth_image[:,:,start_slice:stop_slice],[patch_size_x,patch_size_y,1])
        Truth_patch = Truth_patch.reshape(Truth_patch.shape[0]*Truth_patch.shape[1]*Truth_patch.shape[2],patch_size_x, patch_size_y, 1)
        Truth_patch = Truth_patch[:,(patch_size-1)/2,(patch_size-1)/2]
        
        
        
        
        for i in xrange(1,5):
            #print 'Max : ', max_num_2
            #print 'Present : ', np.sum(image_label==i).astype(int)
            diff = max_num_2-np.sum(image_label==i).astype(int)
            #print 'Diff : ', diff
            if np.sum(image_label==i).astype(int) >= max_num_2:
                #print 'Continuing i = ', i
                continue
            #print 'TEST : ', Truth_patch.shape
            index_x,index_y = np.where(Truth_patch==i)
            #print 'Length : ',len(index_x)
            index = np.arange(len(index_x))
            shuffle(index)
            temp = Truth_patch[index_x[index[0:diff]],:]
            image_label = np.vstack([image_label,temp])
            F_p = Flair_patch[index_x[index[0:diff]],:]
            T1_p = T1_patch[index_x[index[0:diff]],:]
            T2_p = T2_patch[index_x[index[0:diff]],:]
            T_1c_p = T_1c_patch[index_x[index[0:diff]],:]
            temp_patch = np.concatenate([F_p, T1_p, T2_p, T_1c_p], axis=1)
            image_patch = np.vstack([image_patch, temp_patch])
            
            
        
            
#            
#            #print 'image patch : ', image_patch.shape
#            #print 'image_label : ', image_label.shape
#            index_x,index_y = np.where(image_label==i)
#            temp_patch = image_patch[index_x,:]
#            temp_label = image_label[index_x,:]
#            index = np.arange(len(temp_patch))
#            shuffle(index)
#            #print 'Temp patch : ', temp_patch.shape
#            #print 'Temp_label : ', temp_label.shape
#            if len(index)>min_num_2:
#                temp_patch = temp_patch[index[0:min_num_2],:]
#                temp_label = temp_label[index[0:min_num_2],:]
        patches = np.vstack([patches,image_patch])
        ground_truth = np.vstack([ground_truth, image_label])
        for k, item in enumerate(ground_truth):
            if item != 0:
                ground_truth[k] = 1
        
        print 'Number of non-zeros in ground truth : ', np.sum((ground_truth!=0).astype(int))
        print 'Number of zeros in ground truth : ', np.sum((ground_truth==0).astype(int))
    
            
            
            
            
#        patches = np.vstack([patches,slice_patch])
#        ground_truth = np.vstack([ground_truth, Truth_patch])
    print 'Number of non-zeros in ground truth : ', np.sum((ground_truth!=0).astype(int))
    print 'Number of zeros in ground truth : ', np.sum((ground_truth==0).astype(int))
    
    ground_truth = ground_truth.reshape(len(ground_truth))
    
    if recon_flag==False:
        patches = patches[:,0:patch_size*patch_size*4]
    
    if 'training' in out_root and recon_flag == True:
        print'... Saving the 2D training patches'
        np.save(out_root+'b_10_trainpatch_2D_'+prefix+'_.npy',patches)
        np.save(out_root+'b_10_trainlabel_2D_'+prefix+'_.npy',ground_truth)
    elif recon_flag == True:
        print '... Saving the 2D validation patches'
        np.save(out_root+'b_10_validpatch_2D_'+prefix+'_.npy',patches)
        np.save(out_root+'b_10_validlabel_2D_'+prefix+'_.npy',ground_truth)
    
    if 'training' in out_root and recon_flag == False:
        print'... Saving the 2D training patches'
        np.save(out_root+'b_10_trainpatch_2D_'+prefix+'_.npy',patches)
        np.save(out_root+'b_10_trainlabel_2D_'+prefix+'_.npy',ground_truth)
        
    elif recon_flag == False:
        print '... Saving the 2D validation patches'
        np.save(out_root+'b_10_validpatch_2D_'+prefix+'_.npy',patches)
        np.save(out_root+'b_10_validlabel_2D_'+prefix+'_.npy',ground_truth)
