#!/usr/bin/python2.7
import cv2
import os
import numpy as np
from numpy import *

OriginDIR = "bad_cut/"
TargetDIR = "bad_cut_filter_64/"
TargetDIR_dep = "bad_cut_filter_dep_64/"
size = 64
#size = 128
count = 0
quit = 0
limit = size * size / 4
print limit

for image in os.listdir(OriginDIR):
    count = 0
    oriImage = cv2.imread(OriginDIR+image)
    for i in range(size):
        for j in range(size):
            if oriImage[i][j][0]  != 0 or  oriImage[i][j][1] != 0 or  oriImage[i][j][2] != 0 :
                count = count + 1
            if count >=  limit:
                print "copy:"+image
                os.system("cp %s %s" % (OriginDIR+image, TargetDIR+image))
                break;
        if count >=  limit:
            break;
        if i == (size -1 ) and j == (size - 1):
            if count < limit:
                print "Ignore:"+image
                os.system("cp %s %s" % (OriginDIR+image, TargetDIR_dep+image))
            

