# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:42:21 2015

@author: kiran
"""

import itk
import numpy as np
import os

def convert_mha(path,prefix, patchType = 3):
    #print 'Prefix: ',  prefix
    #print 'Path : ', path
    outputs = []
    subdirs = []
    for subdir, dirs, files in os.walk(path):
        #print files
        for file1 in files:
            if prefix in file1 and file1[-3:] == 'npy':
                outputs.append(subdir+'/'+file1)
                subdirs.append(subdir+'/')
                
    
    
    image_type = itk.Image[itk.F,3]
    writer = itk.ImageFileWriter[image_type].New()
    itk_py_converter = itk.PyBuffer[image_type]        
    print 'Entering loop...'
    print 'Number of outputs : ',len(outputs)
    for i in xrange(len(outputs)):
        print 'Saving Image ',i+1
        a=np.load(outputs[i])
        
        a=np.transpose(a)
        if patchType == 2:
            for j in xrange(a.shape[0]):
                a[j,:,:] = np.transpose(a[j,:,:])        
        #a=a.reshape(155,240,240)
        image_type = itk.Image[itk.F,3]
        writer = itk.ImageFileWriter[image_type].New()
        itk_py_converter = itk.PyBuffer[image_type] 
        output_image = itk_py_converter.GetImageFromArray(a.tolist())
        writer.SetFileName(subdirs[i] + prefix+'_.mha')
        writer.SetInput(output_image)
        writer.Update()
        