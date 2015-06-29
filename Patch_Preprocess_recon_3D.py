
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 17:12:53 2015

@author: subru
"""

import mha
import numpy as np
import os
from sklearn.feature_extraction import image
#from matplotlib import pyplot as plt

#Initialize user variables
#patch_size = 125
def Patch_Preprocess_recon_3D(patch_size_x=5,patch_size_y=5,patch_size_z=5,in_root='',out_root='',recon_flag=True):
    
        
    patch_pixels = patch_size_x*patch_size_y*patch_size_z
    
    pixel_offset_x = int(2*patch_size_x*0.7)
    pixel_offset_y = int(2*patch_size_y*0.7)
    pixel_offset_z = 1
    
    padding = patch_size_x/2
    threshold = patch_pixels*0.3
    #patches = np.zeros(patch_pixels*4)
    patches = np.zeros(patch_size_x*patch_size_y*patch_size_z*5)
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
    #    if len(Flair)>5:
    #        break
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
            elif file1[-3:]=='mha' and 'Recon' in file1:
                Recon.append(file1)
                
    number_of_images = len(Flair)
    print 'Number of images : ', number_of_images
    
    
    for image_iterator in range(number_of_images):
        print 'Image number : ',image_iterator+1
        print 'Folder : ', Folder[image_iterator]
        Flair_image = mha.new(Folder[image_iterator]+Flair[image_iterator])
        T1_image = mha.new(Folder[image_iterator]+T1[image_iterator])
        T2_image = mha.new(Folder[image_iterator]+T2[image_iterator])
        T_1c_image = mha.new(Folder[image_iterator]+T_1c[image_iterator])
        Recon_image = mha.new(Folder[image_iterator]+Recon[image_iterator])
        Truth_image = mha.new(Folder[image_iterator]+Truth[image_iterator])
        
        Flair_image = Flair_image.data
        T1_image = T1_image.data
        T2_image = T2_image.data
        T_1c_image = T_1c_image.data
        Recon_image=Recon_image.data
        Truth_image = Truth_image.data
        
        x_span,y_span,z_span = np.where(Truth_image!=0)
        x_start = np.min(x_span) - padding
        x_stop = np.max(x_span) + padding+1
        y_start = np.min(y_span) - padding
        y_stop = np.max(y_span) + padding+1
        z_start = np.min(z_span) - padding
        z_stop = np.max(z_span) +padding+1
        
        Flair_patch = image.extract_patches(Flair_image[x_start:x_stop, y_start:y_stop, z_start:z_stop], [patch_size_x,patch_size_y,patch_size_z],(pixel_offset_x,pixel_offset_y,pixel_offset_z))
        T1_patch = image.extract_patches(T1_image[x_start:x_stop, y_start:y_stop, z_start:z_stop], [patch_size_x,patch_size_y,patch_size_z],(pixel_offset_x,pixel_offset_y,pixel_offset_z))
        T2_patch = image.extract_patches(T2_image[x_start:x_stop, y_start:y_stop, z_start:z_stop], [patch_size_x,patch_size_y,patch_size_z],(pixel_offset_x,pixel_offset_y,pixel_offset_z))
        T_1c_patch = image.extract_patches(T_1c_image[x_start:x_stop, y_start:y_stop, z_start:z_stop], [patch_size_x,patch_size_y,patch_size_z],(pixel_offset_x,pixel_offset_y,pixel_offset_z))
        Recon_patch = image.extract_patches(Recon_image[x_start:x_stop, y_start:y_stop, z_start:z_stop], [patch_size_x,patch_size_y,patch_size_z],(pixel_offset_x,pixel_offset_y,pixel_offset_z))
        Truth_patch = image.extract_patches(Truth_image[x_start:x_stop, y_start:y_stop, z_start:z_stop], [patch_size_x,patch_size_y,patch_size_z],(pixel_offset_x,pixel_offset_y,pixel_offset_z))
        
        print 'Raw patches extracted'    
        
        Flair_patch = Flair_patch.reshape(Flair_patch.shape[0]*Flair_patch.shape[1]*Flair_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)
        T1_patch = T1_patch.reshape(T1_patch.shape[0]*T1_patch.shape[1]*T1_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)
        T2_patch = T2_patch.reshape(T2_patch.shape[0]*T2_patch.shape[1]*T2_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)  
        T_1c_patch = T_1c_patch.reshape(T_1c_patch.shape[0]*T_1c_patch.shape[1]*T_1c_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)
        Recon_patch = Recon_patch.reshape(Recon_patch.shape[0]*Recon_patch.shape[1]*Recon_patch.shape[2], patch_size_x*patch_size_y*patch_size_z)
        Truth_patch = Truth_patch.reshape(Truth_patch.shape[0]*Truth_patch.shape[1]*Truth_patch.shape[2], patch_size_x, patch_size_y, patch_size_z)
    
        print 'Patches reshaped'    
        
        slice_patch = np.concatenate([Flair_patch, T1_patch, T2_patch, T_1c_patch, Recon_patch], axis=1)
        Truth_patch = Truth_patch[:,(patch_size_x-1)/2,(patch_size_y-1)/2,(patch_size_z-1)/2]
        Truth_patch = np.array(Truth_patch)
        Truth_patch = Truth_patch.reshape(len(Truth_patch),1)
        #print '3. truth dimension :', Truth_patch.shape
            
        patches = np.vstack([patches,slice_patch])
        ground_truth = np.vstack([ground_truth, Truth_patch])
    #
    #
    #print 'Number of non-zeros in ground truth : ', np.sum((ground_truth!=0).astype(int))
    #print 'Number of zeros in ground truth : ', np.sum((ground_truth==0).astype(int))
    #
    #print
    #print 'No. of 1 : ', np.sum((ground_truth==1).astype(int))
    #print 'No. of 2 : ', np.sum((ground_truth==2).astype(int))
    #print 'No. of 3 : ', np.sum((ground_truth==3).astype(int))
    #print 'No. of 4 : ', np.sum((ground_truth==4).astype(int))
    #
    ground_truth = ground_truth.reshape(len(ground_truth))
    
    if recon_flag==False:
        patches = patches[:,0:patch_size_x*patch_size_y*patch_size_z*4]
    
    #np.save('Training_patches.npy',patches)
    #np.save('Training_labels.npy',ground_truth)
    #print ground_truth.shape
    #print patches.shape
    if 'training' in out_root and recon_flag==True:
        print'... Saving the training patches'
        np.save(out_root+'trainpatch_3D_'+str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+'_T.npy',patches)
        np.save(out_root+'trainlabel_3D_'+str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+'_T.npy',ground_truth)
    elif recon_flag==True:
        print '... Saving the test ing patches'
        np.save(out_root+'validpatch_3D_'+str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+'_T.npy',patches)
        np.save(out_root+'validlabel_3D_'+str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+'_T.npy',ground_truth)
        
    if 'training' in out_root and recon_flag==False:
        print'... Saving the training patches'
        np.save(out_root+'trainpatch_3D_'+str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+'_F.npy',patches)
        np.save(out_root+'trainlabel_3D_'+str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+'_F.npy',ground_truth)
    elif recon_flag==False:
        print '... Saving the test ing patches'
        np.save(out_root+'validpatch_3D_'+str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+'_F.npy',patches)
        np.save(out_root+'validlabel_3D_'+str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+'_F.npy',ground_truth)
        