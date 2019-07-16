import os
import tensorflow as tf
import itertools as it
import matplotlib.pyplot as plt
import time as t
from random import sample
from utils.csv_utils import write_in_csv
from utils.file_reader import readFile, range_one_input
import numpy as np

batch_types = {0: 'Stochastic Gradient Descent',
               1: 'Mini-Batch Gradient Descent',
               2: 'Batch Gradient Descent'}

mf_types = {0: 'two equations mf',
            1: 'one equation mf'}

class Anfis:
    # num_inputs is not being used. trying the simple way of calculating
    # gradient_type defines the batch size to be used for learning
    # 0-type = Stochastic Gradient Descent
    # 1-type = Mini-Batch Gradient Descent, 30 samples from the training data
    # 2-type = Batch Gradient Descent
    def __init__(self, num_sets, path=None, gradient_type=0, mf_type=0):
        self.num_sets = num_sets
        # self.test_possible = False
        self.train_x_arr = []
        self.train_y_arr = []
        self.test_x_arr = []
        self.test_y_arr = []
        self.constraints = []
        self.gradient_type = gradient_type
        self.full_train = True
        self.mf_type = mf_type

        self.fileName = ''
        with open(path) as file:
            self.fileName = os.path.basename(file.name[:-4])

        # Last index of an array is the expected value.
        # mat array is an array wich contains all the test data.
        if path is not None:
            self.range = range_one_input(path, 0)
            if not self.full_train:
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
                self.train_x_arr = x_arr
                self.train_y_arr = y_arr
                self.test_x_arr = x_arr
                self.test_y_arr = y_arr
                self.num_inputs = len(self.train_x_arr[0])
        else:
            self.num_inputs = 1

        self.num_rules = num_sets ** self.num_inputs

        self.sess = tf.Session()

        # Conclusion Parameters
        self.a_0 = tf.get_variable(name="a_0", dtype=tf.float64,
                                   initializer=np.ones(shape=(self.num_rules, 1))
                                   , trainable=1)

        self.a_y = tf.get_variable(name="a_y", dtype=tf.float64,
                                   initializer=np.ones(shape=(self.num_rules, self.num_inputs))
                                   , trainable=1)

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
        self.premises = [[None] * self.num_rules] * self.num_inputs
        self.second_layer()

        # Third Hidden Layers (Fourth Layer)
        self.normalizedMFs = None
        self.mf_sum = None
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
            self.batch_size = self.num_inputs
            self.x = tf.placeholder(name="x", shape=(self.num_inputs, self.batch_size), dtype=tf.float64)
            # expected result
            self.y = tf.placeholder(name="y", shape=(self.batch_size), dtype=tf.float64)
        elif self.gradient_type == 1:
            self.batch_size = int((1 / 5) * (len(self.train_x_arr)))
            self.x = tf.placeholder(name="x", shape=(self.num_inputs, self.batch_size), dtype=tf.float64)
            # expected result
            self.y = tf.placeholder(name="y", shape=(self.batch_size), dtype=tf.float64)
        elif self.gradient_type == 2:
            self.batch_size = len(self.train_x_arr)
            self.x = tf.placeholder(name="x", shape=(self.num_inputs, self.batch_size), dtype=tf.float64)
            # expected result
            self.y = tf.placeholder(name="y", shape=(self.batch_size), dtype=tf.float64)

        tf.add_to_collection("xVar", self.x)
        tf.add_to_collection("yVar", self.y)

    def triangular_mf(self, x, par, name):
        # we define the triangular function
        if self.mf_type == 0:
            dividend_left = tf.subtract(x, par[0])
            divider_left = tf.subtract(par[1], par[0])
            division_left = tf.divide(dividend_left, divider_left)
            #
            dividend_right = tf.subtract(par[2], x)
            divider_right = tf.subtract(par[2], par[1])
            division_right = tf.divide(dividend_right, divider_right)
            #
            minim = tf.minimum(division_left, division_right)
            maxim = tf.maximum(minim, tf.cast(0.0, tf.float64))
            return maxim
        elif self.mf_type==1:
            dividend = tf.abs(tf.subtract(par[1], x))
            divider = tf.subtract(par[2], par[0])
            op = tf.divide(dividend, divider)
            mul = tf.multiply(tf.cast([2.], tf.float64), op)
            sub = tf.subtract(tf.cast([1.], tf.float64), mul)
            maxim = tf.maximum(tf.cast([0.], tf.float64), sub, name=name)
            return maxim

    def second_layer(self):
        start = t.process_time()
        num_mfs = self.num_sets * self.num_inputs
        m = [[None] * self.num_sets] * self.num_inputs
        val_inc = abs((self.range[1] - self.range[0])) / ((self.num_sets * 3) - 1)

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
                if f > 0:
                    self.constraints.append(tf.assign(a, tf.clip_by_value(a, self.range[0], self.var[i][f - 1][2] - 1)))
                self.premises[i][f] = self.triangular_mf(self.x[i], self.var[i][f], "test_mf" + str(i) + str(f))
            val_arr.clear()
        end = t.process_time()
        print("Initializing Fuzzy Sets took %f s" % (end - start))
        # Creating the Premisses for the rules
        index = 0
        start = t.process_time()
        for perm in it.product(range(self.num_sets), repeat=self.num_inputs):
            tmp = tf.ones(shape=(self.batch_size), dtype=tf.float64)
            for i in range(len(perm)):
                tmp = tf.multiply(self.premises[i][perm[i]], tmp)
            self.mf[index] = tmp
            index += 1
        end = t.process_time()
        print("Initializing all the Permutation took %f s" % (end - start))

    def third_layer(self):
        # Reshape the MFs
        self.mf_sum = tf.reduce_sum(tf.add(self.mf, [1e-12]), 0)

        # Normalize the MFs
        self.normalizedMFs = tf.divide(self.mf, self.mf_sum)

    def define_conclusions(self):
        self.conclusions = tf.add(self.a_0, tf.multiply(self.x, self.a_y), name="outputs")

    def fifth_layer(self):
        self.outputs = tf.multiply(self.normalizedMFs, self.conclusions)

    def sixth_layer(self):
        self.result = tf.reduce_sum(self.outputs, 0)

    def do_calculation(self, sess, x):
        return sess.run(self.result, feed_dict={self.x: [x]})

    def optimize_method(self):
        # 15542508.0
        self.loss = tf.losses.mean_squared_error(self.y, self.result)
        # 15543155.0
        # self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.result))
        # self.loss = tf.losses.huber_loss(self.y, self.result)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1).minimize(self.loss)
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

        # Function before training
        fig, axs = plt.subplots(self.num_inputs * 2 + 1, 1, figsize=(10, 9))
        plt.subplots_adjust(top=0.88, hspace=0.3)

        for i in range(self.num_inputs):
            vars = self.plot_param(sess, axs[i + 1],
                                   "MF %d before training" % (i + 1))
            print("Parameters vor dem Training: {} \n".format(vars[i]))

        if self.gradient_type == 0:
            for i in range(len(self.test_x_arr)):
                x_val.append(self.test_x_arr[i][0])
                y_val.append(self.test_y_arr[i][0])

                y_before_trn.append(self.do_calculation(sess, self.test_x_arr[i]))
        elif self.gradient_type == 2:
            x_val = np.reshape(self.test_x_arr, (len(self.test_x_arr)))
            y_val = np.reshape(self.test_y_arr, (len(self.test_x_arr)))

            y_before_trn = self.do_calculation(sess, x_val)
        else:
            for i in range(5):
                start_index = i * self.batch_size
                end_index = (i + 1) * self.batch_size

                arr_x = self.test_x_arr[start_index:end_index]
                arr_y = self.test_y_arr[start_index:end_index]
                arr_x = np.reshape(arr_x, (self.batch_size))

                x_val.append(arr_x)
                y_val.append(arr_y)

                y_before_trn.append(self.do_calculation(sess, arr_x))

        size = len(self.test_x_arr)
        x_val = np.reshape(x_val, (size))
        y_val = np.reshape(y_val, (size))
        y_before_trn = np.reshape(y_before_trn, (size))

        axs[0].plot(x_val, y_val, color="blue")
        axs[0].plot(x_val, y_before_trn, color="green", alpha=0.5)

        # Training Process
        start = t.process_time()
        if self.gradient_type == 0:
            for s in range(epochs):
                for i in range(len(self.train_x_arr)):

                    sess.run([self.loss, self.optimizer],
                             feed_dict=
                             {self.x: [self.train_x_arr[i]], self.y: self.train_y_arr[i]})
                    sess.run(self.constraints)
        elif self.gradient_type == 1:
            times = 5
            for i in range(epochs):
                for j in range(times):
                    batch_x, batch_y = self.__pick_batch()
                    batch_x = np.reshape(batch_x, (self.batch_size))
                    batch_y = np.reshape(batch_y, (self.batch_size))

                    sess.run([self.loss, self.optimizer],
                             feed_dict={self.x: [batch_x], self.y: batch_y})

                    sess.run(self.constraints)
        else:
            for i in range(epochs):
                size = len(self.train_x_arr)
                x_arr = np.reshape(self.train_x_arr, (self.num_inputs, size))
                y_arr = np.reshape(self.train_y_arr, (size))

                sess.run([self.loss, self.optimizer],
                         feed_dict={self.x: x_arr, self.y: y_arr})
                sess.run(self.constraints)

        end = t.process_time()

        if self.gradient_type == 0:
            for i in range(len(x_val)):
                y_after_trn.append(self.do_calculation(sess, self.test_x_arr[i]))
        elif self.gradient_type == 1:
            times = 5
            for i in range(times):
                start_index = i * self.batch_size
                end_index = (i + 1) * self.batch_size

                arr_x = self.test_x_arr[start_index:end_index]
                arr_x = np.reshape(arr_x, (self.batch_size))
                y_after_trn.append(self.do_calculation(sess, arr_x))
        else:
            size = len(self.test_x_arr)
            x_arr = np.reshape(self.test_x_arr, (size))
            y_after_trn = self.do_calculation(sess, x_arr)

        size = len(self.test_x_arr)
        y_after_trn = np.reshape(y_after_trn, (size))

        # Results after Training
        axs[0].plot(self.test_x_arr, y_after_trn, color='red', alpha=0.5)

        axs[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1,
                      mode="expand", borderaxespad=0.,
                      labels=["Erwartete Werte", "Ergebnisse vor dem Training",
                              "Ergebnisse nach dem Training"])

        for i in range(self.num_inputs):
            vars = self.plot_param(sess, axs[self.num_inputs + 1 + i],
                                   "MF %d after training" % (i + 1))
            print("Parameters nach dem Training: {} \n".format(vars[i]))

        # print(self.do_calculation(sess, [0]))

        prediction = y_after_trn
        labels = y_val
        error = tf.losses.mean_squared_error(labels, prediction)
        v = sess.run(error)

        iterations = epochs

        if self.gradient_type == 0:
            iterations = epochs * len(self.train_x_arr)
        elif self.gradient_type == 1:
            iterations = epochs * 5

        # print(v)
        n = self.fileName + ' 1 Input ' + str(self.num_sets) + ' Sets ' + str(iterations) + ' Epochs ' + batch_types[
            self.gradient_type] + ' ' + mf_types[self.mf_type]

        folder = self.fileName + '/' + batch_types[self.gradient_type].split(' ')[0] + '/'

        fig.savefig('../graphics/fifthgraphics/' + folder +
                    n + '.png')

        row_data = [[n, str(end - start), str(v), batch_types[self.gradient_type], mf_types[self.mf_type]]]
        print(row_data)
        write_in_csv('../csvFiles/model_results.csv', row_data)

        return end - start

    def plot_param(self, sess, plot, name):
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

    def __pick_batch(self):
        indexes = sample(range(len(self.train_x_arr)), self.batch_size)

        batch_x = []
        batch_y = []

        for i in indexes:
            batch_x.append(self.train_x_arr[i][0])
            batch_y.append(self.train_y_arr[i][0])

        return batch_x, batch_y
