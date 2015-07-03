from test_SdA import *
from test_network import *
from testitk import *
import os
import time
from pp2Db import *
from pp2Du import *


if __name__ == '__main__':
    
    root = '../../varghese/10_1_brain_mean/'
    
    patch_size = 5
    
    recon_flag = False
    batch_size = 100
    if recon_flag == True:
        n_ins = patch_size * patch_size * 5
    else:
        n_ins = patch_size * patch_size * 4
    n_outs = 2
    hidden_layers_sizes = [1000,1000,1000]
    corruption_levels = [0.1,0.1,0.1]
    noise_type = 1 #1- Gaussian, 0 - masking
    noise_dict = {1:'Gaussian Noise', 0:'Masking Noise'}
    test_path = root + 'testing'
    prefix = '1or0_5'
    
    print 'Extracting  Balanaced training patches...'
    B_Patch_Preprocess_recon_2D(patch_size,patch_size,prefix,root+'training',
                                      root+'BRATS_training_patches/',False)
    print 'Training patches extracted!'                                      
    
    print 'Extracting  Balanced validation patches...'
    B_Patch_Preprocess_recon_2D(patch_size,patch_size,prefix,root+'validation',                                     
                                      root+'BRATS_validation_patches/',False)
    print 'Validation patches extracted!' 

    print 'Extracting UnBalanced training patches...'
    U_Patch_Preprocess_recon_2D(patch_size,patch_size,prefix,root+'training',
                                      root+'BRATS_training_patches/',False)
    print 'Training patches extracted!'                                      
    
    print 'Extracting Unbalanced validation patches...'
    U_Patch_Preprocess_recon_2D(patch_size,patch_size,prefix,root+'validation',                                     
                                      root+'BRATS_validation_patches/',False)
                                      
    print 'Validation patches extracted!'                                             
                                      
    path = '../results/'
    for subdir, dirs, files in os.walk(path):
        test_num = len(dirs)+1
        break
#        print test_num
        ##########---------SET PREFIX--------##########
    os.mkdir('../results/test'+str(test_num)+'_'+prefix)
#    

    test_root = '../results/test'+str(test_num)+'_'+prefix+'/'
    

    
    print 'Calling test_SdA...'
    
    finetune_lr = 0.1
    pretraining_epochs = 75
    pretrain_lr = 0.001
    training_epochs = 100

    f = open(test_root+prefix+'_params_info.txt', 'w')
    f.write( "Current date & time " + time.strftime("%c"))
    f.write('\nPrefix : '+prefix)
    f.write('\n2D Patches. Patch_size : '+str(patch_size))
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
                root+'BRATS_training_patches/b_10_trainpatch_2D_'+prefix+'_.npy',
                root+'BRATS_training_patches/b_10_trainlabel_2D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/b_10_validpatch_2D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/b_10_validlabel_2D_'+prefix+'_.npy',
                root+'BRATS_training_patches/u_10_trainpatch_2D_'+prefix+'_.npy',
                root+'BRATS_training_patches/u_10_trainlabel_2D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/u_10_validpatch_2D_'+prefix+'_.npy',
                root+'BRATS_validation_patches/u_10_validlabel_2D_'+prefix+'_.npy',batch_size, n_ins, n_outs, hidden_layers_sizes, test_root + prefix, corruption_levels)                
    print 'Network Trained and Saved!'                
                
    test_network(test_root , prefix, test_path, patch_size, patch_size, patch_size, recon_flag, 2)
    
#    convert_mha(root+'testing', prefix, 2) 


    
                

                
