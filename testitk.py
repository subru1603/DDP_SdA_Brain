# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 11:42:21 2015

@author: kiran
"""

import itk
import os
import numpy as np
import time

path = '/home/kiran/Python/'
outputs = []
subdirname = []
for subdir, dirs, files in os.walk(path):
        for file1 in files:
            if 'output' in file1 and 'npy' in subdir:
                outputs.append(subdir+'/'+file1)

image_type = itk.Image[itk.F,3]
writer = itk.ImageFileWriter[image_type].New()
itk_py_converter = itk.PyBuffer[image_type]            

outputPath = 'output/'

start = time.clock()
for i in xrange(len(outputs)):
    print 'Iteration: ',i+1
    a=np.load(outputs[i])
    a=np.transpose(a)
    b = np.zeros(a.shape)
    for j in range(a.shape[0]):
        b[j,:,:] = np.transpose(a[j,:,:])
    print b.shape
    output_image = itk_py_converter.GetImageFromArray(b.tolist())
    writer.SetFileName(outputPath + outputs[i][-23:-4]+'_.mha')
    writer.SetInput(output_image)
    writer.Update()

stop = time.clock()
print 'Time Taken: ', (stop - start)/60.