import tensorflow as tf
import numpy as np
from sklearn import datasets, linear_model

from returns_data import read_goog_sp500_data

xData, yData = read_goog_sp500_data()

# set up a linear model to represent this
googModel = linear_model.LinearRegression()

# reshape converts a single dimantional array to an array of single dimentional arrays
# this is the format in which the sklearn linear regression class requires its input
googModel.fit(xData.reshape(-1, 1), yData.reshape(-1, 1))

# print the coefficients
print(googModel.coef_)
print(googModel.intercept_)

##########################################################################

# sinmple regression - one point per epoch

##########################################################################


# Model liner regression y = Wx + b
W = tf.Variable(tf.zeros([1, 1]))
b = tf.Variable(tf.zeros([1]))

# placeholder to feed in the returns , returns have many rows
# just one column

x = tf.placeholder(tf.float32, [None, 1])

Wx = tf.matmul(x, W)

yPred = Wx + b

# Add summary ops to collect data
# returns a seriallized versiuon of all the values of W
W_hist = tf.summary.histogram("weights", W)
b_hist = tf.summary.histogram("biases", b)

y_pred_hist = tf.summary.histogram("predicted Y", b)

y = tf.placeholder(tf.float32, [None, 1])

# Define the cost
cost = tf.reduce_mean(tf.square(y - yPred))
cost_hist = tf.summary.histogram("cost", cost)

train_step_constant = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
train_step_ada = tf.train.AdagradOptimizer(1).minimize(
    cost)  # this will be worse tan GradeinetDescent
train_step_ftrl = tf.train.FtrlOptimizer(1).minimize(
    cost)  # this will be no better tan GradeinetDescent

# set up a method to perform the actual training, Allows  us to
# modify the optimizer used and also the number of steps
# in the training
dataset_size = len(xData)


def trainWithMultiplePointsPerEpoch(steps, train_step, batch_size):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./linearregression_demo1', sess.graph)

        for i in range(steps):

            if dataset_size == batch_size:
                batch_start_idx = 0
            elif dataset_size < batch_size:
                raise ValueError("dataset size {} must be breater than batch_size {}}".format(
                    dataset_size, batch_size))
            else:
                batch_start_idx = (i * batch_size) % dataset_size

            batch_end_idx = batch_start_idx + batch_size

            batch_xs = xData[batch_start_idx: batch_end_idx]
            batch_ys = yData[batch_start_idx: batch_end_idx]

            feed = {
                x: batch_xs.reshape([-1, 1]),
                y: batch_ys.reshape([-1, 1])
            }
            # Here since the bacth size is sent as the length of the whole xData
            # so the whole dataset is run in every epoch, BUt the batch size parament can be used to
            # reduce the size if desired
           

            sess.run(train_step, feed_dict=feed)

            result = sess.run(merged_summary, feed_dict=feed)
            writer.add_summary(result, i)

            # print result to screen for every 1000 iterations
            if (i + 1) % 1000 == 0:

                print("After {} iterations".format(i))

                print("W: {}".format(sess.run(W)))
                print("b: {}".format(sess.run(b)))

                print("cost: {}".format(sess.run(cost, feed_dict=feed)))

        writer.close()


trainWithMultiplePointsPerEpoch(10000, train_step_ftrl, len(xData))
