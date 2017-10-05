import tensorflow as tf

# Let's load a previously saved meta graph in the default graph
# This function returns a Saver

# We can now access the default graph where all our metadata has been loaded


# Finally we can retrieve tensors, operations, collections, etc.


with tf.Session() as sess:
    # To initialize values with saved data
    saver = tf.train.import_meta_graph('model.ckpt.meta')
    saver.restore(sess, 'model.ckpt.data-00000-of-00001')
