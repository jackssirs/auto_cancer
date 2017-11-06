import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
import collections
import cv2
import os 
import numpy as np
from numpy import *


# Parameters
learning_rate = 1e-3
training_epochs = 15
batch_size = 50
display_step = 1

size = 64
dimension = 1

train_nums = 55335
Train_allcut = "../train/all_cut_64/"

testbatch_size = 41825
Test_allcut = "../test/all_cut_64/"

#testbatch_size = 30720
#Test_allcut = "../test/good_cut_64/"

#testbatch_size = 11105
#Test_allcut = "../test/bad_cut_filter_64/"

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
n_hidden_1 = 256 # 1st layer num features
n_hidden_2 = 256 # 2nd layer num features
n_hidden_3 = 256 # 2nd layer num features
n_input = size * size # MNIST data input (img shape: 28*28)
n_classes = 2 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, size,size])
y = tf.placeholder("float", [None, n_classes])
train_phase = tf.placeholder(tf.bool,name='train_phase')
tst = tf.placeholder(tf.bool)
x_image = tf.reshape(x,[-1,n_input])
keep_prob = tf.placeholder(tf.float32)

def batch_norm_layer(x,train_phase,scope_bn):
    bn_train = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=True,
    reuse=None, # is this right?
    trainable=True,
    scope=scope_bn)
    bn_inference = batch_norm(x, decay=0.999, center=True, scale=True,
    updates_collections=None,
    is_training=False,
    reuse=True, # is this right?
    trainable=True,
    scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

# Create model
def multilayer_perceptron(_X, _weights, _biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with sigmoid activation
    layer_1 = batch_norm_layer(layer_1, train_phase=train_phase, scope_bn='bn0')
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with sigmoid activation
    layer_2 = tf.nn.dropout(layer_2, keep_prob)
    layer_2 = tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3']) #Hidden layer with RELU activation
    layer_2 = batch_norm_layer(layer_2, train_phase=train_phase, scope_bn='bn1')
    layer_3 = tf.nn.relu(layer_2) #Hidden layer with RELU activation
    #layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, _weights['h3']), _biases['b3'])) #Hidden layer with RELU activation
    layer_3 = batch_norm_layer(layer_3, train_phase=train_phase, scope_bn='bn2')
    layer_3 = tf.nn.dropout(layer_3, keep_prob)
    layer_4 = tf.matmul(layer_3, _weights['out']) + _biases['out']
    return layer_4

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_3, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
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
                print ("Step ",i,", Accuracy:", accuracy.eval({x: batch_xs, y: batch_ys, keep_prob: 1, train_phase: False}))
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.5, train_phase: True})
            # Compute average loss
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1, train_phase: False})/total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")


    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    batchtest = get_testbatch()
    #Accuracy: all 0.824
    #Accuracy: good 0.958
    #Accuracy: bad 0.529
    print ("Final Test Accuracy:", accuracy.eval({x: batchtest[0], y: batchtest[1], keep_prob: 1, train_phase: False}))
