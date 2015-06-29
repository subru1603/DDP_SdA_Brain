# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:07:35 2015

@author: bmi
"""

from Patch_Preprocess_recon_3D import *
from test_SdA import *
from test_network import *
from testitk import *
import os


if __name__ == '__main__':
    
    root = '../../varghese/10_1(brain_mean)/'
    
    patch_size_x = 9
    patch_size_y = 9
    patch_size_z = 3
    recon_flag = False
    batch_size = 100
    if recon_flag == True:
        n_ins = patch_size_x * patch_size_y * patch_size_z * 5
    else:
        n_ins = patch_size_x * patch_size_y * patch_size_z * 4
    n_outs = 5
    hidden_layers_sizes = [1000,1000,1000]
    
    test_path = root + 'testing'
    
    print 'Extracting training patches...'
    Patch_Preprocess_recon_3D(patch_size_x,patch_size_y,patch_size_z,root+'training',
                                      root+'BRATS_training_patches/',False)
    print 'Training patches extracted!'                                      
    
    print 'Extracting validation patches...'
    Patch_Preprocess_recon_3D(patch_size_x,patch_size_y,patch_size_z,root+'validation',                                     
                                      root+'BRATS_validation_patches/',False)
    print 'Validation patches extracted!'                                              
                                      
    path = '../results/'
    for subdir, dirs, files in os.walk(path):
        test_num = len(dirs)+1
        break
#        print test_num
        ##########---------SET PREFIX--------##########
    os.mkdir('../results/test'+str(test_num))
    

    test_root = '../results/test'+str(test_num)+'/'
    if recon_flag==True:
        prefix = str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+ '_G4_T_'
        in_name = str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+ '_T'
    else:
        prefix = str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+ '_G4_F_'
        in_name = str(patch_size_x)+'_'+str(patch_size_y)+'_'+str(patch_size_z)+ '_F'
    
    print 'Calling test_SdA...'
    
    finetune_lr = 0.1
    pretraining_epochs = 60
    pretrain_lr = 0.001
    training_epochs = 100
    
    test_SdA(finetune_lr, pretraining_epochs,
             pretrain_lr, training_epochs,              
                root+'BRATS_training_patches/trainpatch_3D_'+in_name+'.npy',
                root+'BRATS_training_patches/trainlabel_3D_'+in_name+'.npy',
                root+'BRATS_validation_patches/validpatch_3D_'+in_name+'.npy',
                root+'BRATS_validation_patches/validlabel_3D_'+in_name+'.npy',batch_size, n_ins, n_outs, hidden_layers_sizes, test_root + prefix)
                
    print 'Network Trained and Saved!'                
                
    test_network(test_root , prefix, test_path, patch_size_x, patch_size_y, patch_size_z, recon_flag)
    
    convert_mha(root+'testing', prefix)
    
    
                

                
