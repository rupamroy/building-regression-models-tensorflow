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

# train_step_constant = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
# train_step_constant = tf.train.AdagradOptimizer(1).minimize(cost) # this will be worse tan GradeinetDescent
train_step_constant = tf.train.FtrlOptimizer(1).minimize(cost) # this will be no better tan GradeinetDescent

# set up a method to perform the actual training, Allows  us to
# modify the optimizer used and also the number of steps
# in the training


def trainWithOnPointPerEpoch(steps, train_step):
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./linearregression_demo1', sess.graph)

        for i in range(steps):

            # Extract one training point this is called stochastic gradient descent
            xs = np.array([[xData[i % len(xData)]]])
            ys = np.array([[yData[i % len(yData)]]])

            feed = {
                x: xs,
                y: ys
            }

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


trainWithOnPointPerEpoch(10000, train_step_constant)
