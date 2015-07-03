# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:14:41 2015

@author: subru
"

# -*- coding: utf-8 -*-
"""


from Patch_Preprocess_recon_3D import *
from test_SdA import *
from test_network import *
from testitk import *
from B_Patch_Preprocess_recon_3D import *
import os
import time


def segment_script(patch_size_x, patch_size_y, patch_size_z, hidden_layers_sizes, corruption_levels, prefix):
    
    root = '../../varghese/10_1_brain_mean/'
    
    #patch_size_x = 9
    #patch_size_y = 9
    #patch_size_z = 9
    recon_flag = False
    batch_size = 100
    
    if recon_flag == True:
        n_ins = patch_size_x * patch_size_y * patch_size_z * 5
    else:
        n_ins = patch_size_x * patch_size_y * patch_size_z * 4
    n_outs = 5
    noise_type = 1
    noise_dict = {1:'Gaussian Noise',0:'Masking Noise'}
    #hidden_layers_sizes = [1000,1000,1000]
    #corruption_levels = [0.01,0.01,0.01]
    
    test_path = root + 'testing'
    
    #prefix = 'xxx'
    
    print 'Extracting  Balanaced training patches...'
    B_Patch_Preprocess_recon_3D(patch_size_x,patch_size_y,patch_size_z,prefix,root+'training',
                                      root+'BRATS_training_patches/',False)
    print 'Training patches extracted!'                                      
    
    print 'Extracting  Balanced validation patches...'
    B_Patch_Preprocess_recon_3D(patch_size_x,patch_size_y,patch_size_z,prefix,root+'validation',                                     
                                      root+'BRATS_validation_patches/',False)
    print 'Validation patches extracted!' 

    print 'Extracting UnBalanced training patches...'
    U_Patch_Preprocess_recon_3D(patch_size_x,patch_size_y,patch_size_z,prefix,root+'training',
                                      root+'BRATS_training_patches/',False)
    print 'Training patches extracted!'                                      
    
    print 'Extracting Unbalanced validation patches...'
    U_Patch_Preprocess_recon_3D(patch_size_x,patch_size_y,patch_size_z,prefix,root+'validation',                                     
                                      root+'BRATS_validation_patches/',False)                                                
                                      
    path = '../results/'
    for subdir, dirs, files in os.walk(path):
        test_num = len(dirs)+1
        break

        
    os.mkdir('../results/test_'+str(test_num)+'_'+prefix)
    test_root = '../results/test_'+str(test_num)+'_'+prefix+'/'
    print 'Calling test_SdA...'
    
    
    finetune_lr = 0.1
    pretraining_epochs = 50
    pretrain_lr = 0.001
    training_epochs = 100

   
    f = open(test_root+prefix+'_params_info.txt', 'w')
    f.write( "Current date & time " + time.strftime("%c"))
    f.write('\nPrefix : '+prefix)
    f.write('\n3D Patches. Patch_size : '+str(patch_size_x)+', '+str(patch_size_y)+', '+str(patch_size_z))
    f.write('\nHidden Layer Sizes : ['+', '.join(map(str,hidden_layers_sizes))+' ]')
    f.write('\nNoise Type : '+noise_dict[noise_type])
    f.write('\nCorruption Levels : ['+', '.join(map(str,corruption_levels))+' ]')
    f.write('\nNo. of pre-training epochs : '+str(pretraining_epochs))
    f.write('\nNo. of Fine-tuning epochs : '+str(training_epochs))
    f.write('\nPretraining Learning rate : '+str(pretrain_lr))
    f.write('\nFine-tuning learning rate : '+str(finetune_lr))
    f.close()
    
    test_SdA(finetune_lr, pretraining_epochs,
             pretrain_lr, training_epochs,              
                root+'BRATS_training_patches/b_trainpatch_3D_'+prefix+'_.npy',
                root+'BRATS_training_patches/b_trainlabel_3D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/b_validpatch_3D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/b_validlabel_3D_'+prefix+'_.npy',
                root+'BRATS_training_patches/u_trainpatch_3D_'+prefix+'_.npy',
                root+'BRATS_training_patches/u_trainlabel_3D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/u_validpatch_3D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/u_validlabel_3D_'+prefix+'_.npy',batch_size, n_ins, n_outs, hidden_layers_sizes, test_root + prefix, corruption_levels)                
    print 'Network Trained and Saved!'                 
                
    test_network(test_root , prefix, test_path, patch_size_x, patch_size_y, patch_size_z, recon_flag)
    
    convert_mha(root+'testing', prefix)
    
    print '####################################################################'
    print
    print 'Completed one network with prefix : ', prefix
    print
    print '####################################################################'
    
    
    
                

                
