#made by Ryan Joshua Liwag


import numpy as np
import argparse
import imutils
import cv2
from math import sqrt

import serial 
import time
import sys
import struct
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



from sklearn.datasets import load_boston    

data=load_boston()

training_epochs = 1000

X_data = data.data
y_data = data.target
m = len(X_data)
n = len(X_data[0])

X = tf.placeholder(tf.float32,[None,n])
y = tf.placeholder(tf.float32,[None,1])

W = tf.Variable(tf.zeros([n,1]))
b = tf.Variable(tf.zeros([1]))

y_ = tf.matmul(X,W)+b

loss = tf.reduce_mean(tf.square( y - y_))

optimizer = tf.train.GradientDescentOptimizer(0.00000001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  #Fit the data
  for epoch in range(training_epochs):
    sess.run(train, feed_dict = {X: X_data, y: y_data[:,None]})


#IMPORT VISION FUNCTION
import vision

# Main function
def main():
	#1 is for webcam another video capture will be here.
	cap = cv2.VideoCapture(1)
	cap.set(3,720)
	cap.set(4,720)
	while(True):

		ret, frame = cap.read()

		cv2.imshow('video', frame)

		#capture replace later with kivy button
		if cv2.waitKey(1) & 0xFF == ord('c'):
			vision.capture(frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break



	cap.release()
	cv2.destroyAllWindows()


if __name__ == '__main__':
	main() 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



from sklearn.datasets import load_boston    

data=load_boston()

training_epochs = 1000

X_data = data.data
y_data = data.target
m = len(X_data)
n = len(X_data[0])

X = tf.placeholder(tf.float32,[None,n])
y = tf.placeholder(tf.float32,[None,1])

W = tf.Variable(tf.zeros([n,1]))
b = tf.Variable(tf.zeros([1]))

y_ = tf.matmul(X,W)+b

loss = tf.reduce_mean(tf.square( y - y_))

optimizer = tf.train.GradientDescentOptimizer(0.00000001)
train = optimizer.minimize(loss)

with tf.Session() as sess:
  init = tf.global_variables_initializer()
  sess.run(init)

  #Fit the data
  for epoch in range(training_epochs):
    sess.run(train, feed_dict = {X: X_data, y: y_data[:,None]})

