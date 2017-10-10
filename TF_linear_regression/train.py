#!/usr/bin/python3

'''
Made by: Ryan Joshua Liwag
'''

from __future__ import print_function, division

import json
import numpy as np
import tensorflow as tf

import pandas as pd
import matplotlib.pyplot as plt

logs_path = 'logs'

def load_data(location):
    data = pd.read_csv(location)
    x_data = data[["Width", "Length", "Thickness"]].values
    y_data = data[["Weight"]].values
    return x_data, y_data

def normalize(data):
    me = data.mean(axis=0)
    std = data.std(axis=0)
    x_data = (data - me) / std
    return x_data


class TFLinearRegression():
    def __init__(self, savefile, D=None, K=None):
        self.savefile = savefile
        if D and K:
            self.build(D,K)

    def build(self, D, K, lr, mu):
        #Linear regression model is y = Wx + b
        self.x = tf.placeholder(tf.float32, [None, D], name = "x")
        self.y_ = tf.placeholder(tf.float32, [None, 1], name = "y_")
        self.W = tf.Variable(tf.zeros([D, 1]), name = "W")
        self.b = tf.Variable(tf.zeros([1]), name = "b")

        self.saver = tf.train.Saver({'W': self.W, 'b': self.b})

        with tf.name_scope("Wx_b") as scope:
            y = tf.add(tf.matmul(self.x, self.W),self.b)

        W_hist = tf.summary.histogram("weights", self.W)
        b_hist = tf.summary.histogram("biases", self.b)
        y_hist = tf.summary.histogram("y", y)

        #cost function (sum((y_-y)^2)
        with tf.name_scope("cost") as scope:
            cost = tf.reduce_mean(tf.square(self.y_-y))
            cost_total = tf.summary.scalar("cost", cost)

        # Gradient Descent
        with tf.name_scope("train") as scope:
            train_op = tf.train.MomentumOptimizer(lr, momentum=mu).minimize(cost)

        #merge all summaries



        return cost, train_op, y

    def fit(self, X, Y, Xtest, Ytest):
        N, D = X.shape
        K = len(Y)

        max_iter = 40000
        lr = 1e-5
        mu = 0.8
        regularization = 1e-1

        cost, train, y = self.build(D, K, lr, mu)
        cost_list = []
        init = tf.global_variables_initializer()
        with tf.Session() as sess:

            sess.run(init)
            merged = tf.summary.merge_all()
            writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            for i in range(max_iter):
                sess.run(train, feed_dict = {self.x: X, self.y_: Y})
                cost_test = sess.run(cost, feed_dict={self.x: X, self.y_:Y})
                y_test = sess.run(y, feed_dict={self.x: X, self.y_:Y})
                print("Cost = ", cost_test)
                cost_list.append(cost_test)
                print("Y = ", y_test)
                result = sess.run(merged, feed_dict= {self.x: X, self.y_: Y})
                writer.add_summary(result, i)

            self.saver.save(sess, self.savefile)

        self.D = D
        self.K = K

    def save(self, filename):
        j = {
            'D': self.D,
            'K': self.K,
            'model': self.savefile
            }
        with open(filename,'w') as f:
            json.dump(j, f)


def main():
    x, y = load_data("dataset/mango_sizes.csv")

    model = TFLinearRegression("tf.model")
    x_norm = normalize(x)
    model.fit(x_norm,y,x,y)

    model.save("my_trained_model.json")



if __name__ == '__main__':
    main()