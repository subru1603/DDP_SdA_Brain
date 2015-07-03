# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 19:18:59 2015

@author: subru
"""

from segment_script import *

patch_size_x = [9,11,7,13]
patch_size_y = [9, 11,7,13]
patch_size_z = [9, 3,3,3]
#noise_type = [1,1,1,1]
hidden_layers_sizes = [[5000,2500,500],[2500,1000,500],[2500,1500,500],[4000,2000,1000,500]]
corruption_levels = [[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1],[0.1,0.1,0.1,0.1]]
prefix = ['9x9x9_5-25-5_G1','11x11x3_25-1-5_G1','7x7x3_25-15-5_G1','13x13x3_4-2-1-5_G1']

for i in xrange(len(patch_size_x)):
    segment_script(patch_size_x[i], patch_size_y[i], patch_size_z[i], hidden_layers_sizes[i], corruption_levels[i], prefix[i])
    
    