from test_SdA import *
import numpy as np
import cPickle
import time
import mha
from sklearn.feature_extraction import image

def initializeNetwork():
    hidden_layers_sizes = [1000,1000,1000]
    
    ins = T.matrix('ins')
    weights = T.matrix('weights')
    biases = T.vector('biases')

    hidden_activation = T.nnet.sigmoid(T.dot(ins,weights) + biases)
    classification = T.nnet.softmax(T.dot(ins,weights) + biases)
    
    activate = theano.function(inputs=[ins,weights,biases], outputs=hidden_activation, allow_input_downcast=True)
    classify = theano.function(inputs = [ins,weights,biases], outputs=classification, allow_input_downcast = True)
    
    weights1 = T.matrix('weights1')
    weights2 = T.matrix('weights2')
    weights3 = T.matrix('weights3')
    weights4 = T.matrix('weights4')
    biases1 = T.vector('biases1')
    biases2 = T.vector('biases2')
    biases3 = T.vector('biases3')
    biases4 = T.vector('biases4')
    layer1 = T.nnet.sigmoid(T.dot(ins,weights1) + biases1)
    layer2 = T.nnet.sigmoid(T.dot(layer1,weights2) + biases2)
    layer3 = T.nnet.sigmoid(T.dot(layer2,weights3) + biases3)
    answer = T.argmax(T.nnet.softmax(T.dot(layer3,weights4) + biases4),axis=1)
    #answer = classify(activate(activate(activate(ins,weights1,biases1),weights2,biases2),weights3,biases3),weights4,biases4)
    activate2 = theano.function(inputs = [ins,weights1,weights2,weights3,weights4,biases1,biases2,biases3,biases4], outputs=answer,allow_input_downcast=True)
    
    openFile_fineTuning = open('SdAfine_tuning.pkl','rb')
    genVariables_load = cPickle.load(openFile_fineTuning)

    W_list = []
    b_list = []

    for i in xrange(len(hidden_layers_sizes) + 1):
        W_list.append(cPickle.load(openFile_fineTuning))
        b_list.append(cPickle.load(openFile_fineTuning))
        
    return activate, classify, activate2, W_list, b_list


def predictOutput(test_data, activate, classify, activate2, W_list, b_list):
    #hidden_layers_sizes = [1000,1000,1000]
    #layers = [activate(test_data,W_list[0],b_list[0])]
    
    #for i in xrange(1, len(hidden_layers_sizes) + 1):
    #    if i < len(hidden_layers_sizes):
    #        layers.append(activate(layers[i-1],W_list[i],b_list[i]))
    #    else:
    #        layers.append(classify(layers[i-1],W_list[i],b_list[i]))
        
    #prediction = layers[-1].argmax(axis=1)
    layers2 = activate2(test_data,W_list[0],W_list[1],W_list[2],W_list[3],b_list[0],b_list[1],b_list[2],b_list[3])
    #prediction = layers2.argmax(axis=1)
#    print 'Softmax layer: ' , layers[-1]
#    print 'Class: ', prediction
        
    return layers2

def classify_test_data(activate, classify, activate2, W_list, b_list):
    path = 'data/Normalized_Testing/'
    Flair = []
    T1 = []
    T2 = []
    T_1c = []
    Folder = []
    patch_size = 11


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
    number_of_images = len(Flair)
    for image_iterator in range(number_of_images):
        print 'Iteration : ',image_iterator+1
        print 'Folder : ', Folder[image_iterator]

        Flair_image = mha.new(Folder[image_iterator]+Flair[image_iterator])
        T1_image = mha.new(Folder[image_iterator]+T1[image_iterator])
        T2_image = mha.new(Folder[image_iterator]+T2[image_iterator])
        T_1c_image = mha.new(Folder[image_iterator]+T_1c[image_iterator])
        Flair_image = Flair_image.data
        T1_image = T1_image.data
        T2_image = T2_image.data
        T_1c_image = T_1c_image.data

        xdim, ydim, zdim = Flair_image.shape
        prediction_image = []
        for i in range(zdim):

            Flair_slice = np.transpose(Flair_image[:,:,i])
            T1_slice = np.transpose(T1_image[:,:,i])
            T2_slice = np.transpose(T2_image[:,:,i])
            T_1c_slice = np.transpose(T_1c_image[:,:,i])

            Flair_patch = image.extract_patches_2d(Flair_slice, (patch_size, patch_size))
            F_P=Flair_patch.reshape(len(Flair_patch),121)
            T1_patch = image.extract_patches_2d(T1_slice, (patch_size, patch_size))
            T1_P=T1_patch.reshape(len(Flair_patch),121)
            T2_patch = image.extract_patches_2d(T2_slice, (patch_size, patch_size))
            T2_P=T2_patch.reshape(len(Flair_patch),121)
            T_1c_patch = image.extract_patches_2d(T_1c_slice, (patch_size, patch_size))
            T1c_P=T_1c_patch.reshape(len(Flair_patch),121)
            num_of_patch = Flair_patch.shape[0]
            #print num_of_patch
            #print Flair_patch.shape

            prediction_slice = []

            #for j in range(num_of_patch):
                #ppre = time.clock()
            temp_patch = np.concatenate([F_P,T1_P,T2_P,T1c_P],axis=1)
            #print temp_patch.shape
                #pre = time.clock()
            prediction_slice = predictOutput(test_data=temp_patch, activate=activate, activate2=activate2, classify=_classify, W_list=_W_list, b_list=_b_list)
            prediction_image.append(prediction_slice)
            #post = time.clock()
                #print 'Time to patch: ' , pre - ppre
                #print 'Time per patch: ' , post - pre
            #np.save('prediction_slice_80_1.npy',prediction_slice)
            
            
            #print 'Length : ',len(prediction_slice)
            #print '230*230 = ', 230*230
        np.save('prediction_image.npy',prediction_image)
        #print prediction_image.shape




        
if __name__ == '__main__':
    
    print 'Loading Network....'
    _activate, _classify, _activate2, _W_list, _b_list = initializeNetwork()
    print 'Network Loaded!'
    
    test_patches = np.load('Test_patches.npy')
    no_of_patches = test_patches.shape[0]
    pred = []
    sum0 = 0
    print 'Predicting outputs....'
    start_time = time.clock()
    #for i in xrange(no_of_patches):
    #    test_data = test_patches[i]
    pred = predictOutput(test_data=test_patches, activate=_activate, activate2=_activate2, classify=_classify, W_list=_W_list, b_list=_b_list)
    #    break
    #print pred
    end_time = time.clock()
    
    print 'Time Taken: ', ((end_time - start_time)/60.)
    pred = np.asarray(pred)
    print pred.shape
    start_time = time.clock()    
    print 'classify_test_data'
    classify_test_data(activate=_activate, classify=_classify, activate2=_activate2, W_list=_W_list, b_list=_b_list)
    end_time = time.clock()
    print 'Classification Time Taken: ', ((end_time - start_time)/60.)
        
        
