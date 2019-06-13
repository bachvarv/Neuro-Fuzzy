import os
import tensorflow as tf
import itertools as it
import matplotlib.pyplot as plt
import time as t
import random
from random import sample
import plotly.plotly as py
import plotly.graph_objs as go
from utils.csv_utils import write_in_csv
from utils.file_reader import readFile, range_one_input
import numpy as np


class Anfis:
    # num_inputs is not being used. trying the simple way of calculating
    # gradient_type defines the batch size to be used for learning
    # 0-type = Stochastic Gradient Descent
    # 1-type = Mini-Batch Gradient Descent, 30 samples from the training data
    # 2-type = Batch Gradient Descent

    # mini_batch_size = 30

    def __init__(self, num_sets, r=None, mat=None, num_inputs=None, path=None, full_train=False, gradient_type=0):
        self.num_sets = num_sets
        # self.test_possible = False
        self.train_x_arr = []
        self.train_y_arr = []
        self.test_x_arr = []
        self.test_y_arr = []
        self.constraints = []
        self.gradient_type = gradient_type
        # self.prediction = []
        # self.labels = []

        self.range = r

        self.fileName = ''
        with open(path) as file:
            self.fileName = os.path.basename(file.name[:-4])

        # Last index of an array is the expected value.
        # mat array is an array wich contains all the test data.
        if mat is not None:
            self.num_inputs = len(mat[0])
            # self.test_possible = True
        elif num_inputs is not None:
            self.num_inputs = num_inputs
        elif path is not None:
            self.range = range_one_input(path, 0)
            print(self.range.dtype)
            # print("Range of value for the variable x1: {}".format(r))
            if not full_train:
                x_arr, y_arr = readFile(path)
                arr_size = len(x_arr)
                train_size = int((3 / 4) * arr_size)
                self.train_x_arr = x_arr[:train_size]
                self.train_y_arr = y_arr[:train_size]
                self.test_x_arr = x_arr[train_size:]
                self.test_y_arr = y_arr[train_size:]
                self.num_inputs = len(self.train_x_arr[0])
            else:
                x_arr, y_arr = readFile(path)

                # print(xArr[500])
                self.train_x_arr = x_arr
                self.train_y_arr = y_arr
                self.test_x_arr = x_arr
                self.test_y_arr = y_arr
                self.num_inputs = len(self.train_x_arr[0])
        else:
            self.num_inputs = 1

        self.num_rules = num_sets ** self.num_inputs

        self.sess = tf.Session()

        # Variables to Create the Membership functions
        # self.premisses = [[None] * self.num_sets]*self.num_inputs
        # self.rules_arr = [None] * self.num_rules

        self.a_0 = tf.get_variable(name="a_0", dtype=tf.float64,
                                   initializer=np.ones(shape=(self.num_rules, 1))
                                   # tf.ones(shape=(self.num_rules, 1))
                                   , trainable=1)

        self.a_y = tf.get_variable(name="a_y", dtype=tf.float64,
                                   initializer=np.ones(shape=(self.num_rules, self.num_inputs))
                                   # tf.ones(shape=(self.num_rules, self.num_inputs))
                                   , trainable=1)

        # self.sess.run(self.a_0.initializer)
        # self.sess.run(self.a_y.initializer)
        #
        self.var = [[None] * self.num_sets] * self.num_inputs

        # Saver to export graph(model)
        self.saver = tf.train.Saver()

        # Input Variable/First Outer Layer (First Layer)
        self.x = None
        # expected result
        self.y = None
        self.first_layer()

        # First and Second Hidden Layer (Second and Third Layer)
        self.mf = [None] * self.num_rules
        # self.member_func(mf_param)
        self.second_layer()

        # Third Hidden Layers (Fourth Layer)
        self.normalizedMFs = None
        self.reshaped_mfs = None
        # self.secondLayer()
        self.third_layer()

        # Definition der Conclusions
        self.conclusions = None
        self.define_conclusions()

        # Fourth Hidden Layer (Fifth Layer)
        self.outputs = None
        self.fifth_layer()

        # Second Outer Layer (Sixth Layer)
        self.result = None
        self.sixth_layer()

        # Optimizer and Loss
        self.optimizer = None
        self.loss = None
        self.optimize_method()

        self.trainableParams = tf.trainable_variables()

    # Finish it with the rest of the program Code

    def first_layer(self):
        if self.gradient_type == 0:
            self.mini_batch_size = self.num_inputs
            self.x = tf.placeholder(name="x", shape=[self.mini_batch_size], dtype=tf.float64)
            tf.add_to_collection("xVar", self.x)
            # expected result
            self.y = tf.placeholder(name="y", shape=(), dtype=tf.float64)
            tf.add_to_collection("yVar", self.y)
        elif self.gradient_type == 1:
            # self.mini_batch_size = int((1 / 5) * (len(self.train_x_arr)))
            self.mini_batch_size = 5
            self.x = tf.placeholder(name="x", shape=[self.mini_batch_size], dtype=tf.float64)
            tf.add_to_collection("xVar", self.x)
            # expected result
            self.y = tf.placeholder(name="y", shape=(self.mini_batch_size), dtype=tf.float64)
            tf.add_to_collection("yVar", self.y)
        elif self.gradient_type == 2:
            self.mini_batch_size = len(self.train_x_arr)
            self.x = tf.placeholder(name="x", shape=[self.mini_batch_size], dtype=tf.float64)
            tf.add_to_collection("xVar", self.x)
            # expected result
            self.y = tf.placeholder(name="y", shape=(self.mini_batch_size), dtype=tf.float64)
            tf.add_to_collection("yVar", self.y)

    @staticmethod
    def triangular_mf(x, par, name):
        # min_left = (x - a1) / (m1 - a1)
        dividend = tf.add(tf.subtract(x, par[0]), 1e-10)
        divider = tf.subtract(par[1], par[0])
        min_left = tf.divide(dividend, divider)

        # min_right = (b1 - x) / (b1 - m1)
        dividend_right = tf.add(tf.subtract(par[2], x), 1e-10)
        divider_right = tf.subtract(par[2], par[1])
        min_right = tf.divide(dividend_right, divider_right)

        min_func = tf.minimum(min_left, min_right)
        m_1 = tf.maximum(min_func, 0.0, name=name)

        return m_1

    def second_layer(self):
        start = t.process_time()
        num_mfs = self.num_sets * self.num_inputs
        premises = [[None] * self.num_sets] * self.num_inputs

        m = [[None] * self.num_sets] * self.num_inputs

        val_inc = abs((self.range[1] - self.range[0])) / ((self.num_sets * 3) - 1)

        # print(abs((self.range[1] - self.range[0])))

        for i in range(self.num_inputs):
            ind = 0
            val = self.range[0]
            val_arr = []
            for f in range(self.num_sets):
                if f == 0:
                    a = tf.get_variable(initializer=tf.constant(self.range[0]),
                                        name="a" + str(i) + str(f),
                                        dtype=tf.float64, trainable=0)
                    val_arr.append(self.range[0])
                else:
                    a = tf.get_variable(initializer=tf.constant((val - 2 * val_inc)),
                                        name="a" + str(i) + str(f),
                                        dtype=tf.float64)
                    val_arr.append((val - 2 * val_inc))

                val = val + val_inc
                ind += 1

                m = tf.get_variable(name="m" + str(i) + str(f), dtype=tf.float64, trainable=1,
                                    initializer=tf.constant(val))

                val_arr.append(val)

                val = val + val_inc

                ind += 1

                if (self.num_sets * i + f) == (num_mfs - 1):
                    b = tf.get_variable(name="b" + str(i) + str(f), dtype=tf.float64,
                                        initializer=tf.constant(self.range[1]), trainable=0)
                    val_arr.append(self.range[1])
                else:
                    b = tf.get_variable(name="b" + str(i) + str(f), dtype=tf.float64,
                                        trainable=1,
                                        initializer=tf.constant(val))

                    val_arr.append(val)

                with tf.variable_scope("", reuse=True):
                    m = tf.get_variable(name="m" + str(i) + str(f), shape=(), dtype=tf.float64,
                                        constraint=lambda x: tf.clip_by_value(m, a, b))

                val = val + val_inc

                self.var[i][f] = a, m, b

                self.constraints.append(tf.assign(a, tf.clip_by_value(a, self.range[0], m - 1e-2)))
                self.constraints.append(tf.assign(b, tf.clip_by_value(b, m + 1e-2, self.range[1])))
                self.constraints.append(tf.assign(m, tf.clip_by_value(m, self.range[0] + 0.1, self.range[1] - 0.1)))

                premises[i][f] = self.triangular_mf(self.x[i], self.var[i][f], "test_mf" + str(i) + str(f))
            val_arr.clear()
        end = t.process_time()
        print("Initializing Fuzzy Sets took %f s" % (end - start))
        # Creating the Premisses for the rules
        index = 0
        start = t.process_time()
        for perm in it.product(range(self.num_sets), repeat=self.num_inputs):
            # num_sets^num_inputs
            # print(perm)
            tmp = tf.ones(shape=(), dtype=tf.float64)
            for i in range(len(perm)):
                tmp = tf.multiply(premises[i][perm[i]], tmp)
            self.mf[index] = tmp
            index += 1
        end = t.process_time()
        print("Initializing all the Permutation took %f s" % (end - start))

    def third_layer(self):
        # Reshape the MFs
        self.reshaped_mfs = tf.tile(self.mf, [self.mini_batch_size])
        # print(self.reshaped_mfs)
        self.reshaped_mfs = tf.reshape(self.reshaped_mfs, shape=(self.num_rules, self.mini_batch_size))
        # print(self.reshaped_mfs)

        # Normalize the MFs
        self.normalizedMFs = tf.divide(self.reshaped_mfs, tf.reduce_sum(self.reshaped_mfs, 0))
        # Older MF
        # Reshape the MFs
        # self.reshaped_mfs = tf.reshape(self.mf, shape=(self.num_rules, 1))
        # Normalize the MFs
        # self.normalizedMFs = tf.divide(self.reshaped_mfs, tf.reduce_sum(self.reshaped_mfs))

    def define_conclusions(self):
        self.conclusions = tf.add(self.a_0, tf.multiply(self.x, self.a_y), name="outputs")

    def fifth_layer(self):
        self.outputs = tf.multiply(self.normalizedMFs, self.conclusions)

    def sixth_layer(self):
        self.result = tf.reduce_sum(self.outputs)

    def do_calculation(self, sess, x):
        return sess.run(self.result, feed_dict={self.x: x})

    def optimize_method(self):
        self.loss = tf.losses.mean_squared_error(self.y, self.result)
        # self.loss = tf.losses.huber_loss(self.y, self.result)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(self.loss)

    @staticmethod
    def get_variable_initializer():
        return tf.global_variables_initializer()

    def save_graph(self, sess, name, step):
        self.saver.save(sess, dir + "/model/" + name, global_step=step)

    def train_2(self, sess, x, y):
        return sess.run([self.loss, self.optimizer], feed_dict={self.x: x, self.y: y})

    @staticmethod
    def all_variables():
        return tf.global_variables()

    def test_model(self, sess, x_data, y_data):
        return sess.run([self.loss, self.result], feed_dict={self.x: x_data, self.y: y_data})

    def train(self, sess, epochs=1):

        sess.run(self.get_variable_initializer())

        # Init of arrays
        x_val = []
        y_val = []
        y_before_trn = []
        y_after_trn = []

        # plt.figure(1, figsize=(8.5, 5))
        # plt.figure(figsize=(8.5, 6))
        fig, axs = plt.subplots(self.num_inputs * 2 + 1, 1, figsize=(10, 9))
        plt.subplots_adjust(top=0.88, hspace=0.3)

        for i in range(self.num_inputs):
            vars = self.plot_param(sess, axs[i + 1],
                                   "MF %d before training" % (i + 1))
            print("Parameters vor dem Training: {} \n".format(vars[i]))

        # plt.subplot((self.num_inputs + 1) * 10 + 1)

        if self.gradient_type == 0:
            for i in range(len(self.test_x_arr)):
                x_val.append(self.test_x_arr[i][0])
                y_val.append(self.test_y_arr[i][0])

                # self.do_calculation(sess, self.test_x_arr[i])
                y_before_trn.append(self.do_calculation(sess, self.test_x_arr[i]))
        elif self.gradient_type == 2:
            x_val = self.test_x_arr
            y_val = self.test_y_arr

            y_before_trn = self.do_calculation(self.do_calculation(sess, self.test_x_arr))
        else:
            iterations = 5
            for i in range(5):
                start_index = i * self.mini_batch_size
                end_index = (i + 1) * self.mini_batch_size
                arr_x = self.test_x_arr[start_index:end_index]
                arr_x = np.reshape(arr_x, (self.mini_batch_size))
                x_val.append(arr_x)
                y_val.append(self.test_y_arr[start_index:end_index])

                y_before_trn.append(self.do_calculation(sess, arr_x))

        # print(y_before_trn)

        # Real Results
        # axs[0].scatter(x_val, y_val, color="blue")
        # plt.plot(x_val, y_val, color="blue")
        # size = len(self.test_x_arr)
        # x_val = np.reshape(x_val, (size))
        # y_val = np.reshape(y_val, (size))
        # print(y_before_trn)
        # y_before_trn = np.reshape(y_before_trn, (size))

        axs[0].plot(x_val, y_val, color="blue")

        # Results before training
        # axs[0].scatter(x_val, y_before_trn, color="green", alpha=0.5)
        # plt.plot(x_val, y_before_trn, color="green", alpha=0.5)
        axs[0].plot(x_val, y_before_trn, color="green", alpha=0.5)
        # Training Process
        start = t.process_time()
        if self.gradient_type == 0:
            for s in range(epochs):
                for i in range(len(self.train_x_arr)):
                    sess.run([self.loss, self.optimizer],
                             feed_dict=
                             {self.x: self.train_x_arr[i], self.y: self.train_y_arr[i][0]})
                    sess.run(self.constraints)
        elif self.gradient_type == 1:
            times = 5
            # times = int(len(self.train_x_arr) / self.mini_batch_size)
            for i in range(epochs):
                for j in range(times):
                    batch_x, batch_y = self.__pick_batch()

                    sess.run([self.loss, self.optimizer],
                             feed_dict={self.x: batch_x, self.y: batch_y})

                    sess.run(self.constraints)
        else:
            for i in range(epochs):
                sess.run([self.loss, self.optimizer],
                         feed_dict={self.x: self.train_x_arr, self.y: self.train_y_arr})

                sess.run(self.constraints)

        end = t.process_time()

        if self.gradient_type == 0:
            for i in range(len(x_val)):
                y_after_trn.append(self.do_calculation(sess, self.test_x_arr[i]))
        elif self.gradient_type == 1:
            times = 5
            for i in range(times):
                start_index = i * self.mini_batch_size
                end_index = (i + 1) * self.mini_batch_size
                arr_x = self.test_x_arr[start_index:end_index]

                y_after_trn.append(self.do_calculation(arr_x))
        else:
            y_after_trn = self.do_calculation(sess, self.test_x_arr)

        size = len(self.test_x_arr)
        # y_after_trn = np.reshape(y_after_trn, (size))

        # Results after Training
        # axs[0].scatter(self.testXArr, y_after_trn, color='red', alpha=0.5)
        # plt.plot(self.testXArr, y_after_trn, color='red', alpha=0.5)
        axs[0].plot(self.test_x_arr, y_after_trn, color='red', alpha=0.5)

        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1,
        #            mode="expand", borderaxespad=0.,
        #            labels=["Erwartete Werte", "Ergebnisse vor dem Training",
        #                    "Ergebnisse nach dem Training"])
        axs[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1,
                      mode="expand", borderaxespad=0.,
                      labels=["Erwartete Werte", "Ergebnisse vor dem Training",
                              "Ergebnisse nach dem Training"])

        for i in range(self.num_inputs):
            vars = self.plot_param(sess, axs[self.num_inputs + 1 + i],
                                   "MF %d after training" % (i + 1))
            print("Parameters nach dem Training: {} \n".format(vars[i]))

        # labels = tf.reshape(tf.constant(y_val), (len(y_val)))
        # predictions = tf.reshape(tf.constant(y_after_trn), (len(y_after_trn)))

        # print(sess.run(labels))
        # print(sess.run(predictions))

        # v = sess.run([self.acc_op, self.acc], feed_dict={self.prediction: y_after_trn, self.labels: y_val})
        # print(v)
        self.prediction = y_after_trn
        self.labels = y_val

        # equality = tf.divide(tf.reduce_sum(tf.abs(tf.subtract(self.labels, self.prediction))),
        #                      len(self.labels))
        equality = tf.losses.mean_squared_error(self.labels, self.prediction)
        # print()
        v = sess.run(equality)
        # v = sess.run(self.accuracy)
        # print(v)
        # sess.run([acc, acc_op])
        # print(sess.run(acc))

        # print(sess.run(self.accuracy))
        n = self.fileName + ' 1 Input ' + str(self.num_sets) + ' Sets ' + str(epochs) + ' Epochs'

        # fig.savefig('../graphics/thirdgraphics/sinus_with_table/' +
        #             n + '.png')

        rowData = [[n, str(end - start), str(v)]]
        print(rowData)
        # write_in_csv('../csvFiles/fnn_data.csv', rowData)
        # self.create_table('1 Input 3 Sets 100 Epochs', end - start, v)

        return end - start

    def test(self, sess):
        for i in range(len(self.test_x_arr)):
            sess.run(self.result, feed_dict={self.x: self.test_x_arr[i], self.y: self.test_y_arr[i][0]})

    # NOTE: This is not needed. Maybe remove it in a later Version.
    def plot_param2(self, sess, ind, name):

        ax1 = plt.subplot(ind)
        mfpar = sess.run(self.var)
        xArr = []
        yArr = []

        ax1.set_title(name)

        for i in mfpar[0]:
            xArr.append(i[0])
            yArr.append(0)
            xArr.append(i[1])
            yArr.append(1)
            xArr.append(i[2])
            yArr.append(0)
            ax1.plot(xArr, yArr)
            # ax1.scatter(xArr, yArr)
            xArr.clear()
            yArr.clear()
        return mfpar

    def plot_param(self, sess, plot, name):

        # ax1 = plt.subplot(ind)
        mfpar = sess.run(self.var)
        xArr = []
        yArr = []

        plot.set_title(name)

        for i in mfpar[0]:
            xArr.append(i[0])
            yArr.append(0)
            xArr.append(i[1])
            yArr.append(1)
            xArr.append(i[2])
            yArr.append(0)
            plot.plot(xArr, yArr)
            xArr.clear()
            yArr.clear()
        return mfpar
        # plt.plot()

    def plot_learning(self, sess):
        x_val = []
        y_val = []
        y_before_trn = []
        y_after_trn = []

        for i in range(len(self.test_x_arr)):
            x_val.append(self.test_x_arr[i])
            y_val.append(self.test_y_arr[i])
            y_before_trn.append(self.do_calculation(sess, self.test_x_arr[i]))

    # NOTE: Maybe not needed
    @staticmethod
    def create_table(type, time, error):

        header = dict(values=['<b>Types</b>', '<b>Time</b>', '<b>Error</b>'],
                      line=dict(color='#7D7F80'),
                      fill=dict(color='#DBDBDB'),
                      align=['left'] * 5)

        cells = dict(values=[[type],
                             [str(time) + 's'],
                             [str(error)]],
                     line=dict(color='#7D7F80'),
                     fill=dict(color='#FFFFFF'),
                     align=['left'] * 5)
        table = go.Table(header=header, cells=cells)

        layout = dict(width=500, height=300)
        data = [table]
        fig = dict(data=data, layout=layout)

        py.plot(fig, filename='fnn_data')
        return fig

    def pick_batch(self, size=0, arr=None):
        if size == 0 and arr == None:
            indexes = sample(range(len(self.train_x_arr)), self.mini_batch_size)

            batch_x = []
            batch_y = []

            for i in indexes:
                batch_x.append(self.train_x_arr[i][0])
                batch_y.append(self.train_y_arr[i][0])


        else:
            indexes = sample(range(len(arr)), size)

            batch_x = []

            batch_y = []

            for i in indexes:
                batch_x.append(arr[i][0])
                batch_y.append(arr[i][0])

        return batch_x, batch_y
