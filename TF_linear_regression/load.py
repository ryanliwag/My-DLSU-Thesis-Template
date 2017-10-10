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


def main():
    X = [[67],[117],[84]]
    with tf.Session() as sess:
      new_saver = tf.train.import_meta_graph('tf.model.meta')
      new_saver.restore(sess, tf.train.latest_checkpoint('./'))

      print(sess.run('W:0'))

if __name__ == '__main__':
    main()