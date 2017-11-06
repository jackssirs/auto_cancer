import tensorflow as tf
import collections
import cv2
import os
import numpy as np
from numpy import *


# Parameters
learning_rate = 1e-4
training_epochs = 5
batch_size = 50
display_step = 1

size = 64
dimension = 1

Train_allcut = "../train/all_cut_64/"
train_nums = 55335

Test_allcut = "../test/all_cut_64/"
testbatch_size = 41825

#Test_allcut = "../test/good_cut_64/"
#testbatch_size = 30720

#Test_allcut = "../test/bad_cut_filter_64/"
#testbatch_size = 11105

print("Initializing...")


#train_images = array([[[[0 for i in range(3)]  for j in range(size)] for k in range(size)] for l in range(batch_size)])
train_images = array([[[0 for j in range(size)] for k in range(size)] for l in range(batch_size)])
test_images = array([[[0 for j in range(size)] for k in range(size)] for l in range(testbatch_size)])
#labels: bad=0,1 good=1,0
train_labels = array([[0 for i in range(2)] for k in range(batch_size)])
test_labels = array([[0 for i in range(2)] for k in range(testbatch_size)])

Datasets = collections.namedtuple('Datasets', ['train_images', 'train_labels'])
DatasetsTest = collections.namedtuple('DatasetsTest', ['test_images', 'test_labels'])

cut_list = []
for img in os.listdir(Train_allcut):
  cut_list.append(img)

testcut_list = []
for img in os.listdir(Test_allcut):
  testcut_list.append(img)

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
    #print(np.shape(train_images[i]))
    #print(train_images[i])
    flag = img[-7:-4]
    if flag == "bad" :
      train_labels[i] = array([0,1])
    else:
      train_labels[i] = array([1,0])

  current = current + batch_realsize

  return Datasets(train_images=train_images, train_labels=train_labels)

def get_testbatch():
  global testbatch_size
  for i in range(testbatch_size):
    img = testcut_list[i]
    test_images[i] = cv2.imread(Test_allcut+img,cv2.IMREAD_GRAYSCALE)
    flag = img[-7:-4]
    if flag == "bad" :
      test_labels[i] = array([0,1])
    else:
      test_labels[i] = array([1,0])

  return DatasetsTest(test_images=test_images, test_labels=test_labels)

# Network Parameters
n_hidden_1 = 2048 # 1st layer num features
n_hidden_2 = 2048 # 2nd layer num features
n_hidden_3 = 2048 # 2nd layer num features
n_hidden_4 = 2048 # 2nd layer num features
n_hidden_5 = 2048 # 2nd layer num features
n_hidden_6 = 2048 # 2nd layer num features
n_hidden_7 = 2048 # 2nd layer num features
n_input = size * size # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, size,size])
y = tf.placeholder("float", [None, n_classes])
tst = tf.placeholder(tf.bool)
x_image = tf.reshape(x,[-1,n_input])
keep_prob = tf.placeholder(tf.float32)

# Create model
def multilayer_perceptron(_X, _weights, _biases):

    layer_1 = tf.nn.tanh(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with sigmoid activation
#    layer_1 = tf.nn.dropout(layer_1, keep_prob)

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with sigmoid activation
#    layer_2 = tf.nn.dropout(layer_2, keep_prob)

    layer_3 = tf.nn.tanh(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])) #Hidden layer with sigmoid activation
#    layer_3 = tf.nn.dropout(layer_3, keep_prob)

    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, _weights['h4']), _biases['b4'])) #Hidden layer with sigmoid activation
#    layer_4 = tf.nn.dropout(layer_4, keep_prob)

    layer_5 = tf.nn.tanh(tf.add(tf.matmul(layer_4, _weights['h5']), _biases['b5'])) #Hidden layer with sigmoid activation
#    layer_5 = tf.nn.dropout(layer_5, keep_prob)

    layer_6 = tf.nn.sigmoid(tf.add(tf.matmul(layer_5, _weights['h6']), _biases['b6'])) #Hidden layer with sigmoid activation
#    layer_6 = tf.nn.dropout(layer_6, keep_prob)

    layer_7 = tf.nn.relu(tf.add(tf.matmul(layer_6, _weights['h7']), _biases['b7'])) #Hidden layer with RELU activation
    layer_7 = tf.nn.dropout(layer_7, keep_prob)

    layer_8 = tf.matmul(layer_7, _weights['out']) + _biases['out']
#    layer_8 = tf.nn.dropout(layer_8, keep_prob)

    return layer_8

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'h7': tf.Variable(tf.random_normal([n_hidden_6, n_hidden_7])),
    'out': tf.Variable(tf.random_normal([n_hidden_7, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'b6': tf.Variable(tf.random_normal([n_hidden_6])),
    'b7': tf.Variable(tf.random_normal([n_hidden_7])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


# Construct model
pred = multilayer_perceptron(x_image, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)) # Softmax loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(train_nums/batch_size )
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = get_next_trainbatch()
            if i% 10 == 0:
                print ("Step ",i,", Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys, keep_prob: 1}))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")


    # Test model
    #correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    batchtest = get_testbatch()
    print ("Final Test Accuracy:", accuracy.eval({x: batchtest[0], y: batchtest[1], keep_prob: 1}))
