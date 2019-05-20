import os
import tensorflow as tf
import itertools as it
import matplotlib.pyplot as plt
import time as t
import random
import plotly.plotly as py
import plotly.graph_objs as go
from utils.csv_utils import write_in_csv
from utils.file_reader import readFile, range_one_input
import numpy as np


class Anfis:
    # num_inputs is not being used. trying the simple way of calculating
    def __init__(self, num_sets, range=None, mat=None, num_inputs=None, path=None, fulltrain=False):
        self.num_sets = num_sets
        # self.test_possible = False
        self.trainXArr = []
        self.trainYArr = []
        self.testXArr = []
        self.testYArr = []
        self.constraints = []
        # self.prediction = []
        # self.labels = []

        self.range = range

        self.fileName = ''
        with open(path) as file:
            self.fileName = os.path.basename(file.name[:-4])

        # Last index of an array is the expected value.
        # mat array is an array wich contains all the test data.
        if (mat != None):
            self.num_inputs = len(mat[0])
            # self.test_possible = True
        elif (num_inputs != None):
            self.num_inputs = num_inputs
        elif path != None:
            self.range = range_one_input(path, 0)
            print(self.range.dtype)
            # print("Range of value for the variable x1: {}".format(r))
            if not fulltrain:
                xArr, yArr = readFile(path)
                sizeOfArr = len(xArr)
                trainSize = int((3 / 4) * sizeOfArr)
                self.trainXArr = xArr[:trainSize]
                self.trainYArr = yArr[:trainSize]
                self.testXArr = xArr[trainSize:]
                self.testYArr = yArr[trainSize:]
                self.num_inputs = len(self.trainXArr[0])
            else:
                xArr, yArr = readFile(path)

                # print(xArr[500])
                self.trainXArr = xArr
                self.trainYArr = yArr
                self.testXArr = xArr
                self.testYArr = yArr
                self.num_inputs = len(self.trainXArr[0])
        else:
            self.num_inputs = 1

        self.num_rules = num_sets ** self.num_inputs

        self.sess = tf.Session()

        # Variables to Create the Membership functions
        # self.premisses = [[None] * self.num_sets]*self.num_inputs
        # self.rules_arr = [None] * self.num_rules

        self.a_0 = tf.get_variable(name="a_0", dtype=tf.float64,
                                   initializer=np.ones(shape=(self.num_rules, 1)).astype(np.float64)
                                   # tf.ones(shape=(self.num_rules, 1))
                                   , trainable=1)

        self.a_y = tf.get_variable(name="a_y", dtype=tf.float64,
                                   initializer=np.ones(shape=(self.num_rules, self.num_inputs)).astype(np.float64)
                                   # tf.ones(shape=(self.num_rules, self.num_inputs))
                                   , trainable=1)

        self.sess.run(self.a_0.initializer)
        self.sess.run(self.a_y.initializer)
        #
        self.var = [[None] * self.num_sets] * self.num_inputs

        # Saver to export graph(model)
        self.saver = tf.train.Saver()

        # input variable
        self.x = tf.placeholder(name="x", shape=[self.num_inputs], dtype=tf.float64)
        tf.add_to_collection("xVar", self.x)
        # expected result
        self.y = tf.placeholder(name="y", shape=(), dtype=tf.float64)
        tf.add_to_collection("yVar", self.y)

        # First Hidden Layer
        self.mf = [None] * self.num_rules
        # self.member_func(mf_param)
        self.mfs()

        # Hidden Layers 2 and 3
        self.normalizedMFs = None
        self.reshaped_mfs = None
        # self.secondLayer()
        self.normLayer_reshaped()

        # Output functions
        self.conclussions = None
        self.defconclussions()

        # Hidden Layer 4
        self.outputs = None
        self.fourthLayer()

        # Fifth Hidden Layer
        self.result = None
        self.fifthLayer()

        # Optimizer and Loss
        self.optimizer = None
        self.loss = None
        self.optimizeMethod()

        self.trainableParams = tf.trainable_variables()

    def triangularMF(self, x, par, name):
        # min_left = (x - a1) / (m1 - a1)
        dividend = tf.add(tf.subtract(x, par[0]), 1e-10)
        dividor = tf.subtract(par[1], par[0])
        min_left = tf.divide(dividend, dividor)

        # min_right = (b1 - x) / (b1 - m1)
        dividend_right = tf.add(tf.subtract(par[2], x), 1e-10)
        dividor_right = tf.subtract(par[2], par[1])
        min_right = tf.divide(dividend_right, dividor_right)

        min_func = tf.minimum(min_left, min_right)
        m_1 = tf.maximum(min_func, 0.0, name=name)

        return m_1

    def normLayer_reshaped(self):
        # Reshape the MFs
        self.reshaped_mfs = tf.reshape(self.mf, shape=(self.num_rules, 1))
        # Normalize the MFs
        self.normalizedMFs = tf.divide(self.reshaped_mfs, tf.reduce_sum(self.reshaped_mfs + 1e-10))

    def defconclussions(self):
        self.conclussions = tf.add(self.a_0, tf.multiply(self.a_y, self.x), name="outputs")

    def outputLayer(self):
        self.summ = tf.reduce_sum(self.outputs + 1e-10)
        self.optimizeMethod()

    def doCalculation(self, sess, x):
        return sess.run(self.result, feed_dict={self.x: x})

    def optimizeMethod(self):
        self.loss = tf.losses.mean_squared_error(self.y, self.result)
        # self.loss = tf.losses.huber_loss(self.y, self.result)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(self.loss)

    def getVariableInitializer(self):
        return tf.global_variables_initializer()

    def save_graph(self, sess, name, step):
        self.saver.save(sess, dir + "/model/" + name, global_step=step)

    def train_2(self, sess, x, y):
        return sess.run([self.loss, self.optimizer], feed_dict={self.x: x, self.y: y})

    def fourthLayer(self):
        self.outputs = tf.multiply(self.normalizedMFs, self.conclussions)

    def fifthLayer(self):
        self.result = tf.reduce_sum(self.outputs + 1e-10)

    # def plot(self):

    def mfs(self):
        start = t.process_time()
        num_mfs = self.num_sets * self.num_inputs
        premisses = [[None] * self.num_sets] * self.num_inputs

        m = [[None] * self.num_sets] * self.num_inputs

        val_inc = abs((self.range[1] - self.range[0])) / ((self.num_sets * 3) - 1)

        print(abs((self.range[1] - self.range[0])))

        ind = 0

        for i in range(self.num_inputs):

            ind = 0
            val = self.range[0]
            valArr = []
            for f in range(self.num_sets):
                if f == 0:
                    a = tf.get_variable(initializer=tf.constant(self.range[0]),
                                        name="a" + str(i) + str(f),
                                        dtype=tf.float64, trainable=0)
                    valArr.append(self.range[0])
                else:
                    a = tf.get_variable(initializer=tf.constant((val - 2 * val_inc)),
                                        name="a" + str(i) + str(f),
                                        dtype=tf.float64)
                    valArr.append((val - 2 * val_inc))

                val = val + val_inc
                ind += 1

                m = tf.get_variable(name="m" + str(i) + str(f), dtype=tf.float64, trainable=1,
                                    initializer=tf.constant(val))

                valArr.append(val)

                val = val + val_inc

                ind += 1

                if (self.num_sets * i + f) == (num_mfs - 1):
                    b = tf.get_variable(name="b" + str(i) + str(f), dtype=tf.float64,
                                        initializer=tf.constant(self.range[1]), trainable=0)
                    valArr.append(self.range[1])
                else:
                    b = tf.get_variable(name="b" + str(i) + str(f), dtype=tf.float64,
                                        trainable=1,
                                        initializer=tf.constant(val))

                    valArr.append(val)

                with tf.variable_scope("", reuse=True):
                    m = tf.get_variable(name="m" + str(i) + str(f), shape=(), dtype=tf.float64,
                                        constraint=lambda x: tf.clip_by_value(m, a, b))

                val = val + val_inc

                self.var[i][f] = a, m, b

                self.constraints.append(tf.assign(a, tf.clip_by_value(a, self.range[0], m - 1e-2)))
                self.constraints.append(tf.assign(b, tf.clip_by_value(b, m + 1e-2, self.range[1])))
                self.constraints.append(tf.assign(m, tf.clip_by_value(m, self.range[0] + 0.1, self.range[1] - 0.1)))

                premisses[i][f] = self.triangularMF(self.x[i], self.var[i][f], "test_mf" + str(i) + str(f))
            valArr.clear()
        end = t.process_time()
        print("Initializing Fuzzy Sets took %f s" % (end - start))
        index = 0
        start = t.process_time()
        for perm in it.product(range(self.num_sets), repeat=self.num_inputs):
            tmp = tf.ones(shape=(), dtype=tf.float64)
            for i in range(len(perm)):
                tmp = tf.multiply(premisses[i][perm[i]], tmp)
            self.mf[index] = tmp
            index += 1
        end = t.process_time()
        print("Initializing all the Permutation takes %f s" % (end - start))

    def all_variables(self):
        return tf.global_variables()

    def testModell(self, sess, x_data, y_data):
        return sess.run([self.loss, self.result], feed_dict={self.x: x_data, self.y: y_data})

    def train(self, sess, it=1):

        sess.run(self.getVariableInitializer())

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
            vars = self.plotParam2(sess, axs[i + 1],
                                   "MF %d before training" % (i + 1))
            print("Parameters vor dem Training: {} \n".format(vars[i]))

        # plt.subplot((self.num_inputs + 1) * 10 + 1)

        for i in range(len(self.testXArr)):
            x_val.append(self.testXArr[i][0])
            y_val.append(self.testYArr[i][0])

            self.doCalculation(sess, self.testXArr[i])
            y_before_trn.append(self.doCalculation(sess, self.testXArr[i]))

        # Real Results
        # axs[0].scatter(x_val, y_val, color="blue")
        # plt.plot(x_val, y_val, color="blue")
        axs[0].plot(x_val, y_val, color="blue")

        # Results before training
        # axs[0].scatter(x_val, y_before_trn, color="green", alpha=0.5)
        # plt.plot(x_val, y_before_trn, color="green", alpha=0.5)
        axs[0].plot(x_val, y_before_trn, color="green", alpha=0.5)
        # Training Process
        start = t.process_time()
        for s in range(it):
            for i in range(len(self.trainXArr)):
                sess.run([self.loss, self.optimizer],
                         feed_dict=
                         {self.x: self.trainXArr[i], self.y: self.trainYArr[i][0]})
                sess.run(self.constraints)

        end = t.process_time()

        for i in range(len(x_val)):
            y_after_trn.append(self.doCalculation(sess, self.testXArr[i]))

        # Results after Training
        # axs[0].scatter(self.testXArr, y_after_trn, color='red', alpha=0.5)
        # plt.plot(self.testXArr, y_after_trn, color='red', alpha=0.5)
        axs[0].plot(self.testXArr, y_after_trn, color='red', alpha=0.5)

        # plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1,
        #            mode="expand", borderaxespad=0.,
        #            labels=["Erwartete Werte", "Ergebnisse vor dem Training",
        #                    "Ergebnisse nach dem Training"])
        axs[0].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1,
                      mode="expand", borderaxespad=0.,
                      labels=["Erwartete Werte", "Ergebnisse vor dem Training",
                              "Ergebnisse nach dem Training"])

        for i in range(self.num_inputs):
            vars = self.plotParam2(sess, axs[self.num_inputs + 1 + i],
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

        equality = tf.divide(tf.reduce_sum(tf.abs(tf.subtract(self.labels, self.prediction))),
                             len(self.labels))
        # print()
        v = sess.run(equality)
        # v = sess.run(self.accuracy)
        # print(v)
        # sess.run([acc, acc_op])
        # print(sess.run(acc))

        # print(sess.run(self.accuracy))
        n = self.fileName + ' 1 Input ' + str(self.num_sets) + ' Sets ' + str(it) + ' Epochs'

        fig.savefig('../graphics/thirdgraphics/sinus_with_table/' +
                    n + '.png')

        rowData = [[n, str(end - start), str(v)]]
        print(rowData)
        write_in_csv('../csvFiles/fnn_data.csv', rowData)
        # self.create_table('1 Input 3 Sets 100 Epochs', end - start, v)

        return end - start

    def test(self, sess):
        for i in range(len(self.testXArr)):
            sess.run(self.result, feed_dict={self.x: self.testXArr[i], self.y: self.testYArr[i][0]})

    def plotMFS(self):
        for i in range(len(self.var)):
            i = 1

    # NOTE: This is not needed. Maybe remove it in a later Version.
    def plotParam(self, sess, ind, name):

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

    def plotParam2(self, sess, plot, name):

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

    def plotLearning(self, sess):
        x_val = []
        y_val = []
        y_before_trn = []
        y_after_trn = []

        for i in range(len(self.testXArr)):
            x_val.append(self.testXArr[i])
            y_val.append(self.testYArr[i])
            y_before_trn.append(self.doCalculation(sess, self.testXArr[i]))

    # NOTE: Maybe not needed
    def create_table(self, type, time, error):

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
