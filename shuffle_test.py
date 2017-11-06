# -*- coding: utf-8 -*-

import os
import numpy as np
import shutil

good_nums = 30720
bad_filter_nums = 11106

train_nums = good_nums + bad_filter_nums

base = np.arange(0,train_nums,1)
np.random.shuffle(base)
Train_good = "test/good_cut_64/"
Train_bad_filter = "test/bad_cut_filter_64/"
Train_all = "test/all_cut_64/"

i = 0
for img in os.listdir(Train_good):
  print i
  shutil.copyfile(Train_good+img,Train_all+str(base[i])+"_"+img)
  print (Train_all+str(base[i])+"_"+img)
  i = i + 1

for img in os.listdir(Train_bad_filter):
  print i
  shutil.copyfile(Train_bad_filter+img,Train_all+str(base[i])+"_"+img+".bad.png")
  print (Train_all+str(base[i])+"_"+img+".bad.png")
  i = i + 1

print "Total shuffle:"+str(i-1)
