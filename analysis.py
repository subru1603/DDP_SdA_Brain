import os
import getopt
import sys

ANTSPATH = '/usr/local/antsbin/bin/'
LABELPATH = '/media/UUI/BRATS_Outputs/Patients'
TRUTHPATH = '/media/UUI/BRATS_Outputs/Patients'

files = os.listdir(LABELPATH)
LABELPATH, patients, files = os.walk(LABELPATH).next()

for p in patients:
    print p
    imgs=os.listdir(LABELPATH+'/'+p)
    for i in imgs:
            if '506070' in i:
                classified_path=LABELPATH+'/'+p+'/'+i
            if 'OT' in i:
                truth_path=TRUTHPATH+'/'+p+'/'+i
    print classified_path
    print truth_path
    os.system(ANTSPATH+'LabelOverlapMeasures 3 ' +truth_path+' '+classified_path+' 1')