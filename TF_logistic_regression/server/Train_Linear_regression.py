# tensorflow implementation of multiple linear regression or logistic regression
# Train Thickness, lenght, width to obtain weight

import tensorflow as tf 
import numpy
from sklearn.datasets import load_boston

randoms = numpy.random

data = load_boston()
X_data = data.data
y_data = data.target


lr = 0.00001
training_epochs = 1000
display_step = 50

nb_samples = len(X_data)

nb_features = len(X_data[0])

#tf Graph input
X = tf.placeholder(tf.float32, [None, nb_features])
Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([nb_features,1]), name = "Weights")
b = tf.Variable(tf.zeros([1]), name = "Bias")

pred = tf.add(tf.matmul(X,W), b)

cost = tf.reduce_mean(tf.square(pred - Y))

optimizer = tf.train.GradientDescentOptimizer(lr)
train = optimizer.minimize(cost)
#THis initialize all the variables(W,b)

saver = tf.train.Saver()

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  #Fit the data
  for epoch in range(training_epochs):
    sess.run(train, feed_dict = {X: X_data, Y: y_data[:,None]})

  save_path = saver.save(sess,"model.ckpt")











            sess.run(optimizer, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})

            if step % FLAGS.display_step == 0:
                print "Step:", "%04d" % (step+1), "Cost=", "{:.2f}".format(sess.run(cost, \
                    feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})), "W=", sess.run(W), "b=", sess.run(b)

        print "Optimization Finished!"
        training_cost = sess.run(cost, feed_dict={X: np.asarray(train_X), Y: np.asarray(train_Y)})
        print "Training Cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n'

        print "Predict.... (Predict a house with 1650 square feet and 3 bedrooms.)"
        predict_X = np.array([1650, 3], dtype=np.float32).reshape((1, 2))

        # Do not forget to normalize your features when you make this prediction
        predict_X = predict_X / np.linalg.norm(predict_X)

        predict_Y = tf.add(tf.matmul(predict_X, W),b)
        print "House price(Y) =", sess.run(predict_Y)

