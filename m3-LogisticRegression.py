import pandas as pd
import numpy as np
import statsmodels.api as sm


from returns_data import read_google_sp500_logistic_data

xData, yData = read_google_sp500_logistic_data()

logit = sm.Logit(yData, xData)

#Fit the logistic model to fit to an S-curve
results = logit.fit()

#All values > 0.5 predict an up day for Google
predictions = (results.predict(xData) > 0.5)

# Count the number of times the actual up days match the predicted up days
num_accurate_predictions = (list(yData == predictions)).count(True)

pctAccuracy = float(num_accurate_predictions) / float(len(predictions))

print("Accuracy: {}".format(pctAccuracy))

###########################################
#
# Reimplement logistic regression in tensorflow
#
##############################################
import tensorflow as tf

W = tf.Variable(tf.ones([1,2]), name="W")

b = tf.Variable(tf.zeros([2]), name="b")

x = tf.placeholder(tf.float32, [None, 1], name="x") # will hold [n x 1] martix
y_ = tf.placeholder(tf.float32, [None, 2], name = "y_") # will hold [n x 2] matrix , this will hold 1 hot representation of data

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)) # this will give us the cost function which needs to be minimised

# Add summary ops to collect data
# returns a seriallized versiuon of all the values of W
W_hist = tf.summary.histogram("weights", W)
b_hist = tf.summary.histogram("biases", b)

y_hist = tf.summary.histogram("predicted Y", b)

cost_hist = tf.summary.histogram("cost", cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# the below function call to expand_dims returns a 2D array
# [[-0.02184618]
#  [0.00997998]
#  [0.04329069]
#  [0.03254923]
#  [0.01701632]]
all_xs = np.expand_dims(xData[:,0], axis=1) # [:,0] returns only the 0th col , we do not require the Incerpt column since that was required only for the statsmodel

# We change our y to return true/false data in 2D array in a 1-hot format
# 2D aray with 0 1 or 1 0 in each row
# 1 0 indicates a UP day
# 0 1 indicates a DOWN day
# [[0 1]
# [1 0]
# [1 0]
# [1 0]
# [0 1]
# [1 0]]
all_ys=np.array([([1,0] if yEl is True else [0,1]) for yEl in yData])

dataset_size = len(all_xs)

def trainWithMultiplePointsPerEpoch(steps, train_step, batch_size):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./linearregression_demo2', sess.graph)

        for i in range(steps):

            if dataset_size == batch_size:
                batch_start_idx = 0
            elif dataset_size < batch_size:
                raise ValueError("dataset size {} must be breater than batch_size {}".format(dataset_size, batch_size))
            else: 
                batch_start_idx = (i * batch_size) % dataset_size

            batch_end_idx = batch_start_idx + batch_size

            batch_xs = all_xs[batch_start_idx: batch_end_idx]
            batch_ys = all_ys[batch_start_idx: batch_end_idx]


            feed = {
                x: batch_xs,
                y: batch_ys
            }

            sess.run(train_step, feed_dict=feed)

            result = sess.run(merged_summary, feed_dict=feed)
            writer.add_summary(result, i)

            # print result to screen for every 1000 iterations
            if (i + 1) % 1000 == 0:

                print("After {} iterations".format(i))

                print("W: {}".format(sess.run(W)))
                print("b: {}".format(sess.run(b)))

                print("cross _entropy: {}".format(sess.run(cross_entropy, feed_dict=feed)))

        writer.close()
        #Test model
        correct_prediction = tf.equal(tf.argmax(y_,1),tf.argmax(y,1))

        # calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        print("Accuracy {}".format(sess.run(accuracy, feed_dict={x: all_xs, y:all_ys})))


trainWithMultiplePointsPerEpoch(1000, train_step, 10)
    

