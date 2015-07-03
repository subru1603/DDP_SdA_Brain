# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:35:46 2015

@author: bmi
"""

from Patch_Preprocess_recon_2D import *
from test_SdA import *
from test_network import *
from testitk import *
import os

def segment_2D_script(patch_size, hidden_layers_sizes, corruption_levels, prefix):
    
    root = '../../varghese/10_1_brain_mean/'
    
#    patch_size = 11
    
    recon_flag = False
    batch_size = 100
    if recon_flag == True:
        n_ins = patch_size * patch_size * 5
    else:
        n_ins = patch_size * patch_size * 4
    n_outs = 5
#    hidden_layers_sizes = [1000,1000,1000]
#    corruption_levels = [0.01,0.01,0.01]
#    noise_type = 1 #1- Gaussian, 0 - masking
    
    test_path = root + 'testing'
#    prefix = 'yyy'
    
    print 'Extracting training patches...'
    Patch_Preprocess_recon_2D(patch_size,patch_size, prefix,root+'training',root+'BRATS_training_patches/',False)
    print 'Training patches extracted!'                                      
    
    print 'Extracting validation patches...'
    Patch_Preprocess_recon_2D(patch_size,patch_size,prefix,root+'validation',                                     
                                      root+'BRATS_validation_patches/',False)
    print 'Validation patches extracted!'                                              
                                      
    path = '../results/'
    for subdir, dirs, files in os.walk(path):
        test_num = len(dirs)+1
        break
#        print test_num
        ##########---------SET PREFIX--------##########
    os.mkdir('../results/test2D'+str(test_num)+'_'+prefix)
#    

    test_root = '../results/test2D'+str(test_num)+'_'+prefix+'/'
    

    
    print 'Calling test_SdA...'
    
    finetune_lr = 0.1
    pretraining_epochs = 1
    pretrain_lr = 0.001
    training_epochs = 1
    
    test_SdA(finetune_lr, pretraining_epochs,
             pretrain_lr, training_epochs,              
                root+'BRATS_training_patches/trainpatch_2D_'+prefix+'_.npy',
                root+'BRATS_training_patches/trainlabel_2D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/validpatch_2D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/validlabel_2D_'+prefix+'_.npy',batch_size, n_ins, n_outs, hidden_layers_sizes, test_root + prefix, corruption_levels)
                
    print 'Network Trained and Saved!'                
                
    test_network(test_root , prefix, test_path, patch_size, patch_size, patch_size, recon_flag, 2)
    
    convert_mha(root+'testing', prefix, 2) 
#    
#    