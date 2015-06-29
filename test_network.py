import theano.tensor as T
import theano
import numpy as np
import cPickle
import time
import mha
from sklearn.feature_extraction import image
import os

def initializeNetwork(prefix):
    
    openFile_fineTuning = open(prefix+'fine_tuning.pkl','rb')
    hidden_layers_sizes = cPickle.load(openFile_fineTuning)
    
    ins = T.matrix('ins')
    
    theano_weights = []
    theano_biases = []
    theano_layers = []
    for i in xrange(len(hidden_layers_sizes) + 1):
        theano_weights.append(T.matrix('weights'))
        theano_biases.append(T.vector('biases'))
        if i is 0:
            theano_layers.append(T.nnet.sigmoid(T.dot(ins,theano_weights[i]) + theano_biases[i]))
        elif i is len(hidden_layers_sizes):
            theano_layers.append(T.nnet.softmax(T.dot(theano_layers[i-1],theano_weights[i]) + theano_biases[i]))
        else:
            theano_layers.append(T.nnet.sigmoid(T.dot(theano_layers[i-1],theano_weights[i]) + theano_biases[i]))
        
    answer = T.argmax(theano_layers[len(hidden_layers_sizes)],axis=1)
    
    print ins
    print theano_weights
    print theano_biases
    
    _input = [ins] + theano_weights + theano_biases
    
    activate2 = theano.function(inputs = _input, outputs=answer,allow_input_downcast=True)
    
    
    genVariables_load = cPickle.load(openFile_fineTuning)

    W_list = []
    b_list = []

    for i in xrange(len(hidden_layers_sizes) + 1):
        W_list.append(cPickle.load(openFile_fineTuning))
        b_list.append(cPickle.load(openFile_fineTuning))
        
    return activate2, W_list, b_list


def predictOutput(test_data, activate2, W_list, b_list):
    '''To predict output from the given test image'''
    layers2 = activate2(test_data,W_list[0],W_list[1],W_list[2],W_list[3],b_list[0],b_list[1],b_list[2],b_list[3])
    return layers2

def classify_test_data(activate2, W_list, b_list, path, patch_size, prefix, recon_flag):
    Flair = []
    T1 = []
    T2 = []
    T_1c = []
    Recon = []
    Folder = []
    Subdir_array = []

    for subdir, dirs, files in os.walk(path):
        for file1 in files:
            #print file1
            if file1[-3:]=='mha' and 'Flair' in file1:
                Flair.append(file1)
                Folder.append(subdir+'/')
                Subdir_array.append(subdir[-5:])
            elif file1[-3:]=='mha' and 'T1' in file1:
                T1.append(file1)
            elif file1[-3:]=='mha' and 'T2' in file1:
                T2.append(file1)
            elif file1[-3:]=='mha' and 'T_1c' in file1:
                T_1c.append(file1)
            elif file1[-3:]=='mha' and 'Recon' in file1:
                Recon.append(file1)
    number_of_images = len(Flair)
    
    for image_iterator in range(number_of_images):
        print 'Iteration : ',image_iterator+1
        print 'Folder : ', Folder[image_iterator]
        print '... predicting'

        Flair_image = mha.new(Folder[image_iterator]+Flair[image_iterator])
        T1_image = mha.new(Folder[image_iterator]+T1[image_iterator])
        T2_image = mha.new(Folder[image_iterator]+T2[image_iterator])
        T_1c_image = mha.new(Folder[image_iterator]+T_1c[image_iterator])
        Recon_image = mha.new(Folder[image_iterator]+Recon[image_iterator])
        Flair_image = Flair_image.data
        T1_image = T1_image.data
        T2_image = T2_image.data
        T_1c_image = T_1c_image.data
        Recon_image = Recon_image.data

        xdim, ydim, zdim = Flair_image.shape
        prediction_image = []
        
        for i in range(zdim):
            Flair_slice = np.transpose(Flair_image[:,:,i])
            T1_slice = np.transpose(T1_image[:,:,i])
            T2_slice = np.transpose(T2_image[:,:,i])
            T_1c_slice = np.transpose(T_1c_image[:,:,i])
            Recon_slice = np.transpose(Recon_image[:,:,i])

            Flair_patch = image.extract_patches_2d(Flair_slice, (patch_size, patch_size))
            F_P=Flair_patch.reshape(len(Flair_patch),patch_size*patch_size)
            T1_patch = image.extract_patches_2d(T1_slice, (patch_size, patch_size))
            T1_P=T1_patch.reshape(len(Flair_patch),patch_size*patch_size)
            T2_patch = image.extract_patches_2d(T2_slice, (patch_size, patch_size))
            T2_P=T2_patch.reshape(len(Flair_patch),patch_size*patch_size)
            T_1c_patch = image.extract_patches_2d(T_1c_slice, (patch_size, patch_size))
            T1c_P=T_1c_patch.reshape(len(Flair_patch),patch_size*patch_size)
            Recon_patch = image.extract_patches_2d(Recon_slice, (patch_size, patch_size))
            R_P=Recon_patch.reshape(len(Flair_patch),patch_size*patch_size)
            
            if recon_flag == True:            
                temp_patch = np.concatenate([F_P,T1_P,T2_P,T1c_P,R_P],axis=1)
            else:
                temp_patch = np.concatenate([F_P,T1_P,T2_P,T1c_P],axis=1)
        
            prediction_slice = predictOutput(temp_patch, activate2, W_list, b_list)
            prediction_image.append(prediction_slice)
        
        prediction_image = np.array(prediction_image)
        prediction_image = np.transpose(prediction_image)
        prediction_image = prediction_image.reshape([xdim-patch_size+1, ydim-patch_size+1, zdim])
        output_image = np.zeros([xdim,ydim,zdim])
        output_image[1+((patch_size-1)/2):xdim-((patch_size-1)/2)+1,1+((patch_size-1)/2):ydim-((patch_size-1)/2)+1,:] = prediction_image      
        np.save(Folder[image_iterator]+Subdir_array[image_iterator]+'_'+prefix+'_output_image.npy',output_image)#TODO: save it in meaningful name in corresponding folder
        

def classify_test_data_3d(activate2, W_list, b_list, path, patch_size_x, patch_size_y, patch_size_z, prefix, recon_flag):
    
    Flair = []
    T1 = []
    T2 = []
    T_1c = []
    Folder = []
    Recon = []
    Subdir_array = []
#    patch_size = 11

    for subdir, dirs, files in os.walk(path):
        for file1 in files:
            #print file1
            if file1[-3:]=='mha' and 'Flair' in file1:
                Flair.append(file1)
                Folder.append(subdir+'/')
                Subdir_array.append(subdir[-5:])
            elif file1[-3:]=='mha' and 'T1' in file1:
                T1.append(file1)
            elif file1[-3:]=='mha' and 'T2' in file1:
                T2.append(file1)
            elif file1[-3:]=='mha' and 'T_1c' in file1:
                T_1c.append(file1)
            elif file1[-3:]=='mha' and 'Recon' in file1:
                Recon.append(file1)
    number_of_images = len(Flair)
    
    for image_iterator in range(number_of_images):
        print 'Iteration : ',image_iterator+1
        print 'Folder : ', Folder[image_iterator]
        print '... predicting'

        Flair_image = mha.new(Folder[image_iterator]+Flair[image_iterator])
        T1_image = mha.new(Folder[image_iterator]+T1[image_iterator])
        T2_image = mha.new(Folder[image_iterator]+T2[image_iterator])
        T_1c_image = mha.new(Folder[image_iterator]+T_1c[image_iterator])
        Recon_image = mha.new(Folder[image_iterator]+Recon[image_iterator])
        Flair_image = Flair_image.data
        T1_image = T1_image.data
        T2_image = T2_image.data
        T_1c_image = T_1c_image.data
        Recon_image = Recon_image.data

        xdim, ydim, zdim = Flair_image.shape
        prediction_image = []
        Flair_patch = image.extract_patches(Flair_image, [patch_size_x,patch_size_y,patch_size_z])
        T1_patch = image.extract_patches(T1_image, [patch_size_x,patch_size_y,patch_size_z])
        T2_patch = image.extract_patches(T2_image, [patch_size_x,patch_size_y,patch_size_z])
        T_1c_patch = image.extract_patches(T_1c_image, [patch_size_x,patch_size_y,patch_size_z])
        Recon_patch = image.extract_patches(Recon_image, [patch_size_x,patch_size_y,patch_size_z])
        
        print 'Raw patches extracted'
        #print Flair_patch.shape
        #print T1_patch.shape
        #print T2_patch.shape
        #print T_1c_patch.shape
        
        for j in range(Flair_patch.shape[2]):
            #print 'Slice : ',j+1
            F_slice = Flair_patch[:,:,j,:,:,:]
            T1_slice = T1_patch[:,:,j,:,:,:]
            T2_slice = T2_patch[:,:,j,:,:,:]
            T_1c_slice = T_1c_patch[:,:,j,:,:,:]
            Recon_slice = Recon_patch[:,:,j,:,:,:]
            
            F_slice = F_slice.reshape(F_slice.shape[0]*F_slice.shape[1], patch_size_x*patch_size_y*patch_size_z)
            T1_slice = T1_slice.reshape(T1_slice.shape[0]*T1_slice.shape[1], patch_size_x*patch_size_y*patch_size_z)
            T2_slice = T2_slice.reshape(T2_slice.shape[0]*T2_slice.shape[1], patch_size_x*patch_size_y*patch_size_z)
            T_1c_slice = T_1c_slice.reshape(T_1c_slice.shape[0]*T_1c_slice.shape[1], patch_size_x*patch_size_y*patch_size_z)
            Recon_slice = Recon_slice.reshape(Recon_slice.shape[0]*Recon_slice.shape[1], patch_size_x*patch_size_y*patch_size_z)
            
            if recon_flag == True:
                temp_patch = np.concatenate([F_slice,T1_slice,T2_slice,T_1c_slice,Recon_slice],axis=1)
            else:
                temp_patch = np.concatenate([F_slice,T1_slice,T2_slice,T_1c_slice],axis=1)
            #print 'Size of temp_patch : ',temp_patch.shape
            prediction_slice = predictOutput(temp_patch, activate2, W_list, b_list)
            prediction_image.append(prediction_slice)
            
        prediction_image = np.array(prediction_image)
        prediction_image = np.transpose(prediction_image)
        prediction_image = prediction_image.reshape([xdim-patch_size_x+1, ydim-patch_size_y+1, zdim-patch_size_z+1])
        output_image = np.zeros([xdim,ydim,zdim])
        output_image[1+((patch_size_x-1)/2):xdim-((patch_size_x-1)/2)+1,1+((patch_size_y-1)/2):ydim-((patch_size_y-1)/2)+1,1+((patch_size_z-1)/2):zdim-((patch_size_z-1)/2)+1] = prediction_image      
        np.save(Folder[image_iterator]+Subdir_array[image_iterator]+'_'+prefix+'_output_image.npy',output_image)#TODO: save it in meaningful name in corresponding folder
        
        
#        
#        
#        prediction_image = np.array(prediction_image)
#        prediction_image = np.transpose(prediction_image)
#        prediction_image = prediction_image.reshape([xdim-patch_size+1, ydim-patch_size+1, zdim])
#        output_image = np.zeros([xdim,ydim,zdim])
#        output_image[1+((patch_size-1)/2):xdim-((patch_size-1)/2)+1,1+((patch_size-1)/2):ydim-((patch_size-1)/2)+1,:] = prediction_image      
#        np.save(Folder[image_iterator]+Subdir_array[image_iterator]+'_output_image.npy',output_image)#TODO: save it in meaningful name in corresponding folder
#

        
def test_network(root,prefix, path, patch_size_x, patch_size_y, patch_size_z, recon_flag, patch_shape=3):
    
    print 'Loading Network....'
    activate2, W_list, b_list = initializeNetwork(root+prefix)
    print 'Network Loaded!'
    
    pred = []
    print 'Predicting outputs....'
    start_time = time.clock()    
    print 'classify_test_data'
    if patch_shape == 2:
        classify_test_data(activate2, W_list,b_list,path,patch_size_x, prefix,recon_flag)
    else:
        classify_test_data_3d(activate2, W_list, b_list, path, patch_size_x, patch_size_y, patch_size_z, prefix, recon_flag)
    end_time = time.clock()
    print 'Classification Time Taken: ', ((end_time - start_time)/60.)
        