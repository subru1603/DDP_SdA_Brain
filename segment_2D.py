# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 11:07:35 2015

@author: bmi
"""

from Patch_Preprocess_recon_2D import *
from test_SdA import *
from test_network import *
from testitk import *
import os


if __name__ == '__main__':
    
    root = '../../varghese/10_1(brain_mean)/'
    
    patch_size = 11
    
    recon_flag = False
    batch_size = 100
    if recon_flag == True:
        n_ins = patch_size * patch_size * 5
    else:
        n_ins = patch_size * patch_size * 4
    n_outs = 5
    hidden_layers_sizes = [1000,1000,1000]
    
    test_path = root + 'testing'
    
    print 'Extracting training patches...'
    Patch_Preprocess_recon_2D(patch_size,patch_size,root+'training',
                                      root+'BRATS_training_patches/',False)
    print 'Training patches extracted!'                                      
    
    print 'Extracting validation patches...'
    Patch_Preprocess_recon_2D(patch_size,patch_size,root+'validation',                                     
                                      root+'BRATS_validation_patches/',False)
    print 'Validation patches extracted!'                                              
                                      
    path = '../results/'
    for subdir, dirs, files in os.walk(path):
        test_num = len(dirs)+1
        break
#        print test_num
        ##########---------SET PREFIX--------##########
    os.mkdir('../results/test'+str(test_num))
#    

    test_root = '../results/test'+str(test_num)+'/'
    if recon_flag==True:
        prefix = str(patch_size)+'_'+str(patch_size)+'_G4_T'
        in_name = str(patch_size)+'_'+str(patch_size)+'_T'
    else:
        prefix = str(patch_size)+'_'+str(patch_size)+'_G4_F'
        in_name = str(patch_size)+'_'+str(patch_size)+'_F'
    
    print 'Calling test_SdA...'
    
    finetune_lr = 0.1
    pretraining_epochs = 1
    pretrain_lr = 0.001
    training_epochs = 1
    
    test_SdA(finetune_lr, pretraining_epochs,
             pretrain_lr, training_epochs,              
                root+'BRATS_training_patches/trainpatch_2D_'+in_name+'.npy',
                root+'BRATS_training_patches/trainlabel_2D_'+in_name+'.npy',
                root+'BRATS_validation_patches/validpatch_2D_'+in_name+'.npy',
                root+'BRATS_validation_patches/validlabel_2D_'+in_name+'.npy',batch_size, n_ins, n_outs, hidden_layers_sizes, test_root + prefix)
                
    print 'Network Trained and Saved!'                
                
    test_network(test_root , prefix, test_path, patch_size, patch_size, patch_size, recon_flag, 2)
    
    convert_mha(root+'testing', prefix, 2) 
    
    
                

                
