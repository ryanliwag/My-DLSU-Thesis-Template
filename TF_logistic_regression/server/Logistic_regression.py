'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

from sklearn.datasets import load_iris
import pandas as pd


data = load_iris()
X_data = data.data
y_data = data.target

nb_samples = len(X_data)

nb_features = len(X_data[0] )
nb_classes = len(data.target_names)


S = pd.Series(list(y_data))
g = pd.get_dummies(S)

def chunks1(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 10
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [150, 4]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, nb_classes]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([nb_features, nb_classes]))
b = tf.Variable(tf.zeros([nb_classes]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)

saver = tf.train.Saver()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(tf.global_variables_initializer())
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        _, c = sess.run([optimizer, cost], feed_dict={x: X_data, y: g})


        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    save_path = saver.save(sess, "test/model.ckpt")