import string, os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
import tensorflow as tf

# set the folder path
dir_name = 'train/all_cut_64/'

# set the file path
files = os.listdir(dir_name)

file_path = dir_name + os.sep+files[14]

# get the data
dic_mat = scipy.io.loadmat(file_path)

data_mat = dic_mat['Hog_Feat']

print (dic_mat.keys())

print (type(dic_mat))

print ('feature: ',  data_mat.shape)

# print data_mat.dtype
file_path2 = dir_name + os.sep + files[15]

# print file_path2
dic_label = scipy.io.loadmat(file_path2)

label_mat = dic_label['Label']
file_path3 = dir_name + os.sep+files[16]
print ('file 3 path: ', file_path3)
dic_T = scipy.io.loadmat(file_path3)

T = dic_T['T']
T = T-1

print (T.shape)

label = label_mat.ravel()
print (label.shape)
label_y = np.zeros((4000, 2))
label_y[:, 0] = label
label_y[:, 1] = 1-label

print (label_y.shape)
T_ind=random.sample(range(0, 4000), 4000)

# Parameters
learning_rate = 0.005
train_epoch=100
batch_size = 40
batch_num=4000/batch_size

# Network Parameters
n_hidden_1 = 300 # 1st layer number of features
n_hidden_2 = 100 # 2nd layer number of features
n_input = 1764 # data input 
n_classes = 2 # total classes (2)
drop_out = 0.5


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Create some wrappers for simplicity
# 定义MLP 函数
def multilayer_perceptron(x, weights, biases, drop_out):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # layer_2 = tf.nn.dropout(layer_2, drop_out)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['w_out']) + biases['b_out']
    return out_layer

# Store layers weight & bias
weights = {
    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'w_out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}


biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b_out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases, drop_out)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables
init = tf.initialize_all_variables()

train_loss = np.zeros(train_epoch)
train_acc = np.zeros(train_epoch)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(0, train_epoch):
        for batch in range (0, batch_num):
            arr_3 = T_ind[ batch * batch_size : (batch + 1) * batch_size ]
            batch_x = data_mat[arr_3, :]
            batch_y = label_y[arr_3, :]
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # Calculate loss and accuracy
        loss, acc = sess.run([cost, accuracy], feed_dict={x: data_mat,
                                                          y: label_y})


        train_loss[epoch] = loss
        train_acc[epoch] = acc


        print("Epoch: " + str(epoch+1) + ", Loss= " + \
                  "{:.3f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

plt.subplot(211)
plt.plot(train_loss, 'r')
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.grid(True)


plt.subplot(212)
plt.xlabel("Epoch")
plt.ylabel('Training Accuracy')
plt.ylim(0.0, 1)
plt.plot(train_acc, 'r')
plt.grid(True)

plt.show()
