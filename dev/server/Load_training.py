import tensorflow as tf

# Let's load a previously saved meta graph in the default graph
# This function returns a Saver

# We can now access the default graph where all our metadata has been loaded


# Finally we can retrieve tensors, operations, collections, etc.

saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess, "model.ckpt")
    # To initialize values with saved data
	sess.run(tf.global_variables_initializer())
    # Training cycle
	for epoch in range(training_epochs):
	    avg_cost = 0.
	    _, c = sess.run([optimizer, cost], feed_dict={x: X_data, y: g})


	    if (epoch+1) % display_step == 0:
	        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

	print("Optimization Finished!")
