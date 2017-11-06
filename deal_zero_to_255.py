import collections
import cv2
import os
import numpy as np
from numpy import *


# Parameters
size = 64

train_nums = 55335
Train_allcut = "./train/all_cut_64/"
Test_allcut = "./test/all_cut_64/"
Test_badcut = "./test/bad_cut_filter_64/"
img = [[[0 for i in range(3)]  for j in range(size)]  for k in range(size)]

def zero_to_white(imagedir):
  no = 0
  for image in os.listdir(imagedir):
    no = no + 1
    img = cv2.imread(imagedir+image)
    for i in range(size):
      for j in range(size):
        if img[i][j][0] == 0 and img[i][j][1] == 0 and img[i][j][2] == 0:
          img[i][j][0] == 255
          img[i][j][0] == 255
          img[i][j][0] == 255
    cv2.imwrite(imagedir+image,img)
    print str(no)+":"+imagedir+image

zero_to_white(Train_allcut)
zero_to_white(Test_allcut)
zero_to_white(Test_badcut)
