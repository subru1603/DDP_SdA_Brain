# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 13:25:07 2015

@author: bmi
"""

from test_network import *

#prefix = ['11x11_1000-500-250-100_G1','11x11_2000-1000-1000-500_G1','13x13_1000-500-250_G1','13x13_2000-1000-500-100_G1','9x9_2000-1000-500-100_G1',
#            '9x9_1000-500-250-100_G1','9x9_1000-1000-1000_G1','9x9_1000-500-100_G1','7x7_1000-500-250_G1']

prefix = ['Tumor_NonTumor_9x9x9_5000-200-500G1','Tumor_NonTumor_11x11x11_5000-2000-500G15']

root = '../../varghese/10_1_brain_mean/'
#test_root = ['/home/bmi/BRATS/results/test2D9_11x11_1000-500-250-100_G1/',
#             '/home/bmi/BRATS/results/test2D10_11x11_2000-1000-1000-500_G1/',
#             '/home/bmi/BRATS/results/test2D11_13x13_1000-500-250_G1/',
#             '/home/bmi/BRATS/results/test2D12_13x13_2000-1000-500-100_G1/',
#             '/home/bmi/BRATS/results/test2D13_9x9_2000-1000-500-100_G1/',
#             '/home/bmi/BRATS/results/test2D14_9x9_1000-500-250-100_G1/',
#             '/home/bmi/BRATS/results/test2D15_9x9_1000-1000-1000_G1/',
#             '/home/bmi/BRATS/results/test2D16_9x9_1000-500-100_G1/',
#             '/home/bmi/BRATS/results/test2D17_7x7_1000-500-250_G1/']

test_root = ['/home/bmi/BRATS/results/test17_Tumor_NonTumor_9x9x9_5000-200-500G1/',
             '/home/bmi/BRATS/results/test18_Tumor_NonTumor_9x9x9_5000-2000-500G15/']
             
patch_z = [11,11,13,13,9,9,9,9,7]
             
test_path = root+'testing'

for i in xrange(len(prefix)):
    test_network(test_root[i],prefix[i],test_path,9,9,9,False,3)
    print i