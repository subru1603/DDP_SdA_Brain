# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 21:39:10 2015

@author: bmi
"""

from segment_2D_script import *

patch_size = [11,11,13,13,9,9,9,9]

noise_type = [1,1]

hidden_layers_sizes = [[1000,500,250,100],[1000,1000,1000,500],[1000,500,250],[1000,500,250,100],[1000,1000,1000,1000],[1000,500,250,100],[1000,500,250],[1000,1000,500]]
c3 = [0.01,0.01,0.01]
c4 = [0.01,0.01,0.01,0.01]
corruption_levels = [c4,c4,c3,c4,c4,c4,c3,c3]
prefix = ['11x11-4l','11x11-4l_2','13x13-3l','13x13-4l','9x9-4l','9x9-4l_2','9x9-3l','9x9-3l_2']

for i in xrange(len(patch_size)):
    segment_2D_script(patch_size[i], hidden_layers_sizes[i], corruption_levels[i], prefix[i])