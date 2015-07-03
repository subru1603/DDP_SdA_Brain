import os
import getopt
import sys


prefix = 'misclass'
ANTSPATH = '/home/bmi/antsbin/bin/'
LABELPATH = '/home/bmi/varghese/10_1_brain_mean/testing'
TRUTHPATH = '/home/bmi/varghese/10_1_brain_mean/testing'

files = os.listdir(LABELPATH)
LABELPATH, patients, files = os.walk(LABELPATH).next()
#print patients
for p in patients:
#    print p
    imgs=os.listdir(LABELPATH+'/'+p)
    for i in imgs:
            if prefix in i and  i[-3:]=='mha':
                classified_path=LABELPATH+'/'+p+'/'+i
            if 'OT' in i :
                truth_path=TRUTHPATH+'/'+p+'/'+i
            
#    print classified_path
#    print truth_path
#    print(ANTSPATH+'LabelOverlapMeasures 3 ' +truth_path+' '+classified_path+' 1')
    os.system(ANTSPATH+'LabelOverlapMeasures 3 ' +truth_path+' '+classified_path+' ')
