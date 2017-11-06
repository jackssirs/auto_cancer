# -*- coding: utf-8 -*-

#use AdamOptimizer (1-e4)
import collections
import tensorflow as tf 
import cv2
import os
import numpy as np
from numpy import *

size = 64
train_nums = 210082
#test_nums = 41826
test_nums = 4182
Train_allcut = "train/all_cut_64/"
Test_allcut = "test/all_cut_64/"

#Train_good = "train/good_cut/"
#Train_bad = "train/bad_cut_filter/"
#Test_good = "test/good_cut/"
#Test_bad = "test/bad_cut_filter/"
print("Initializing...")

batch_size = 50
train_images = array([[[[0 for i in range(3)]  for j in range(size)] for k in range(size)] for l in range(batch_size)])
#test_images = array([[[[0 for i in range(3)]  for j in range(size)] for k in range(size)] for l in range(test_nums)])
#labels: bad=0,1 good=1,0
train_labels = array([[0 for i in range(2)] for k in range(batch_size)])
#test_labels = array([[0 for i in range(2)] for k in range(test_nums)])

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
    train_images[i] = cv2.imread(Train_allcut+img)
    flag = img[-7:-4]
    if flag == "bad" :
      train_labels[i] = array([0,1])
    else:
      train_labels[i] = array([1,0])
  current = current + batch_realsize

  return Datasets(train_images=train_images, train_labels=train_labels)

#def get_testbatch():
#  global test_nums
#  for i in range(test_nums):
#    img = cut_list[current+i]
#    train_images[i] = cv2.imread(Train_allcut+img)
#    flag = img[-7:-4]
#    if flag == "bad" :
#      train_labels[i] = array([0,1])
#    else: 
#      train_labels[i] = array([1,0])
#  current = current + batch_realsize
#
#  return Datasets(train_images=train_images, train_labels=train_labels)

x = tf.placeholder(tf.float32, [None, 64,64,3])                        #输入的数据占位符 64x64x3
y_actual = tf.placeholder(tf.float32, shape=[None, 2])            #输入的标签占位符
keep_prob = tf.placeholder("float") 

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
x_image = tf.reshape(x,[-1,64,64,3])         #转换输入数据shape,以便于用于网络中 64x64x3 -> 60x60x3

W_conv1 = weight_variable([5, 5, 3, 64])      
b_conv1 = bias_variable([64])       
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)     #第一个卷积层 60x60x64
h_pool1 = max_pool(h_conv1)                                  #第一个池化层 30x30x64

W_conv2 = weight_variable([5, 5, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      #第二个卷积层 30x30x128
h_pool2 = max_pool(h_conv2)                                   #第二个池化层 16x16x128

W_fc1 = weight_variable([16 * 16 * 128, 512])
b_fc1 = bias_variable([512])
h_pool3_flat = tf.reshape(h_pool2, [-1, 16*16*128])              #reshape成向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)    #第一个全连接层

h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                  #dropout层

W_fc2 = weight_variable([512, 2])
b_fc2 = bias_variable([2])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   #softmax层

cross_entropy = -tf.reduce_sum(y_actual*tf.log(y_predict))     #交叉熵
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)    #梯度下降法
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)    #梯度下降法
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #精确度计算
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())

for i in range(int(train_nums/batch_size) + 1):
  batch = get_next_trainbatch()
  if i%100 == 0:                  #训练100次，验证一次
    train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
    print('step',i,'training accuracy',train_acc)
    pass
  train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})

exit(0)
test_acc=accuracy.eval(feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
print("Final test accuracy",test_acc)
