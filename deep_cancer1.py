# -*- coding: utf-8 -*-
# modify filter size -> 20x20
# modify batch size -> 50

import collections
import tensorflow as tf 
import cv2
import os
import numpy as np
from numpy import *

size = 128
train_nums = 55335
#train_nums_good = 112640
#train_nums_bad = 97442

Train_allcut = "train/all_cut/"
#Train_good = "train/good_cut/"
#Train_bad = "train/bad_cut_filter/"
#Test_good = "test/good_cut/"
#Test_bad = "test/bad_cut_filter/"
print("Initializing...")

batch_size = 20
#train_images = array([[[[0 for i in range(3)]  for j in range(size)] for k in range(size)] for l in range(batch_size)])
train_images = array([[[0 for j in range(size)] for k in range(size)] for l in range(batch_size)])
#labels: bad=0,1 good=1,0
train_labels = array([[0 for i in range(2)] for k in range(batch_size)])

Datasets = collections.namedtuple('Datasets', ['train_images', 'train_labels'])

cut_list = []
for img in os.listdir(Train_allcut):
  cut_list.append(img)

current = 0
def get_next_trainbatch():
  global batch_size
  global train_nums
  global current
  batch_realsize = batch_size
  if (train_nums - current) <= batch_size:
    batch_realsize = (train_nums - current)
  for i in range(batch_realsize):
    img = cut_list[current+i]
    train_images[i] = cv2.imread(Train_allcut+img,cv2.IMREAD_GRAYSCALE)
    flag = img[-7:-4]
    #print ("------")
    #print (img)
    if flag == "bad" :
      train_labels[i] = array([0,1])
    else:
      train_labels[i] = array([1,0])
    #print (train_labels[i])

  current = current + batch_realsize

  return Datasets(train_images=train_images, train_labels=train_labels)

x = tf.placeholder(tf.float32, [None, size , size])                        #输入的数据占位符 64x64x3
y_actual = tf.placeholder(tf.float32, shape=[None, 2])            #输入的标签占位符

#定义一个函数，用于初始化所有的权值 W
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

#定义一个函数，用于初始化所有的偏置项 b
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
#定义一个函数，用于构建卷积层
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#定义一个函数，用于构建池化层
def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

#构建网络
x_image = tf.reshape(x,[-1,size,size,1])         #转换输入数据shape,以便于用于网络中 64x64x3 -> 60x60x3

W_conv1 = weight_variable([5, 5, 1, 128])      
b_conv1 = bias_variable([128])       
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     #第一个卷积层 60x60x64
h_pool1 = max_pool(h_conv1)                                  #第一个池化层 30x30x64

W_conv2 = weight_variable([5, 5, 128, 256])
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #第二个卷积层 30x30x128
h_pool2 = max_pool(h_conv2)                                   #第二个池化层 16x16x128

W_fc1 = weight_variable([32 * 32 * 256, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 32*32*256])              #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)    #第一个全连接层

keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax层

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #交叉熵
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)    #梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #精确度计算
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())
#print(h_fc1_drop.get_shape())
#print(W_fc2.get_shape())
#print(b_fc2.get_shape())
#print(y_predict.get_shape())
#exit(0)

#train_flag=1
for i in range(int(train_nums/batch_size) + 1):
  batch = get_next_trainbatch()
  #print (np.shape(batch[0]))
  #print (np.shape(batch[1]))
  #exit(0)
  if i%10 == 0:                  #训练100次，验证一次
    train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
    print('Traing step',i,'training accuracy',train_acc)
  train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

exit(0)

#  if i%100 == 0 and train_flag == 1:                  #训练100次，验证一次
#    train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
#    print('Traing step',i,'training accuracy',train_acc)
#  else:
#    train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
#    print('Test step',i,'training accuracy',train_acc)
#    train_flag=0

#  if train_acc < 0.9 and train_flag == 1:
#    train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})
#  else:
#    train_flag = 0


test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print("Final test accuracy",test_acc)
