from test_SdA import *
import numpy as np
import cPickle
import time

def initializeNetwork():
    hidden_layers_sizes = [1000,1000,1000]
    
    ins = T.vector('ins')
    weights = T.matrix('weights')
    biases = T.vector('biases')

    hidden_activation = T.nnet.sigmoid(T.dot(ins,weights) + biases)
    classification = T.nnet.softmax(T.dot(ins,weights) + biases)
    
    activate = theano.function(inputs=[ins,weights,biases], outputs=hidden_activation, allow_input_downcast=True)
    classify = theano.function(inputs = [ins,weights,biases], outputs=classification, allow_input_downcast = True)

    openFile_fineTuning = open('SdAfine_tuning.pkl','rb')
    genVariables_load = cPickle.load(openFile_fineTuning)

    W_list = []
    b_list = []

    for i in xrange(len(hidden_layers_sizes) + 1):
        W_list.append(cPickle.load(openFile_fineTuning))
        b_list.append(cPickle.load(openFile_fineTuning))
        
    return activate, classify, W_list, b_list


def predictOutput(test_data, activate, classify, W_list, b_list):
    hidden_layers_sizes = [1000,1000,1000]
    layers = [activate(test_data,W_list[0],b_list[0])]
    
    for i in xrange(1, len(hidden_layers_sizes) + 1):
        if i < len(hidden_layers_sizes):
            layers.append(activate(layers[i-1],W_list[i],b_list[i]))
        else:
            layers.append(classify(layers[i-1],W_list[i],b_list[i]))
        
    prediction = layers[-1].argmax(axis=0)
        
#    print 'Softmax layer: ' , layers[-1]
#    print 'Class: ', prediction
        
    return prediction
        
        
if __name__ == '__main__':
    
    print 'Loading Network....'
    _activate, _classify, _W_list, _b_list = initializeNetwork()
    print 'Network Loaded!'
    
    test_patches = np.load('Test_patches.npy')
    no_of_patches = test_patches.shape[0]
    pred = []
    
    print 'Predicting outputs....'
    start_time = time.clock()
    for i in xrange(no_of_patches):
        test_data = test_patches[i]
        pred.append(predictOutput(test_data=test_data, activate=_activate, classify=_classify, W_list=_W_list, b_list=_b_list))
    end_time = time.clock()
    
    print 'Time Taken: ', ((end_time - start_time)/60.)
        
        
        
    
    

    