#!/usr/bin/python2.7
import cv2
import os
import numpy as np
from numpy import *

OriginDIR = "bad/"
TargetDIR = "bad_cut/"
size = 64
num = 2048/size
count = 0

smallimg = array([[[0 for i in range(3)] for j in range(size)] for k in range(size)])
zeroimg = array([[[0 for i in range(3)] for j in range(size)] for k in range(size)])

for image in os.listdir(OriginDIR):
    oriImage = cv2.imread(OriginDIR+image)
    for i in range(num):
        for j in range(num):
            saveimage = image+"."+str(i)+"_"+str(j)+".png"
            for px in range(size):
                for py in range(size):
                    smallimg[px][py][0] = oriImage[i*size + px][j*size + py][0]
                    smallimg[px][py][1] = oriImage[i*size + px][j*size + py][1]
                    smallimg[px][py][2] = oriImage[i*size + px][j*size + py][2]
            if(smallimg == zeroimg).all():
                print "Ignore: "+TargetDIR+saveimage
                continue
            cv2.imwrite(TargetDIR+saveimage,smallimg)
            print TargetDIR+saveimage

    count = count + 1

print "Total:"+str(count)
