import os

import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

class Anfis:
    # num_inputs is not being used. trying the simple way of calculating
    def __init__(self, num_inputs, num_rules, mf_param=None):
        self.num_rules = num_rules

        self.a_0 = tf.get_variable(name="a_0", dtype=tf.float32, initializer=tf.ones(shape=(num_rules, 1)), trainable=1)
        self.a_y = tf.get_variable(name="a_y", dtype=tf.float32,
                                   initializer=tf.ones(shape=(num_rules, num_inputs)), trainable=1)

        size = len(mf_param)
        #
        self.var = [None] * size

        # Saver to export graph(model)
        self.saver = tf.train.Saver()

        # for i in range(num_inputs):
        #     self.x = tf.placeholder(name="x"+str(i + 1), shape=(num_inputs), dtype=tf.float32)

        # input variable
        self.x = tf.placeholder(name="x", shape=(num_inputs), dtype=tf.float32)
        tf.add_to_collection("xVar", self.x)
        # expected result
        self.y = tf.placeholder(name="y", shape=(), dtype=tf.float32)
        tf.add_to_collection("yVar", self.y)

        # self.mfParameters = mf_param
        # print(len(self.mfParameters))

        # First Hidden Layer
        self.mf = [None]*(size)
        self.member_func(mf_param)

        # Hidden Layers 2 and 3
        self.normalizedMFs = None
        self.reshaped_mfs = None
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

        # index = 1
        # for i in mf_param:
        #     # a = None
        #     # m = None
        #     # b = None
        #     # with tf.variable_scope("variables", reuse=tf.AUTO_REUSE):
        #     a = tf.get_variable(name="a"+str(index), dtype=tf.float32, initializer=tf.constant(i[0]), trainable=1)
        #     m = tf.get_variable(name="m"+str(index), dtype=tf.float32, initializer=tf.constant(i[1]), trainable=1)
        #     b = tf.get_variable(name="b"+str(index), dtype=tf.float32, initializer=tf.constant(i[2]), trainable=1)
        #     self.var[index - 1] = [a, m, b]
        #
        #     with tf.variable_scope("mfs", reuse=tf.AUTO_REUSE):
        #         self.mf[(index - 1)] = self.triangularMF(self.x, a, m, b, ("mf" + str(index)))
        #
        #     index += 1

        # self.result = tf.

        # output layer selbst erzeugen am Ende
        #self.outputLayer(num_rules, num_inputs)

        # self.optimizeMethod()

        # FIXME:
        # self.conc_func(num_rules)
        self.trainableParams = tf.trainable_variables()
        # self.init = tf.global_variables_initializer()

    def triangularMF(self, x, a, m, b, name):
        # min_left = (x - a1) / (m1 - a1)
        # dividend = tf.abs(tf.subtract(x, a))
        dividend = tf.subtract(x, a)
        dividor = tf.subtract(m, a)
        min_left = tf.divide(dividend, dividor)
        # min_left = tf.subtract(1.0, tf.multiply(2.0, tf.divide(dividend, dividor)))

        # min_right = (b1 - x) / (b1 - m1)
        # dividend_right = tf.abs(tf.subtract(b, x))
        dividend_right = tf.subtract(b, x)
        dividor_right = tf.subtract(b, m)

        min_right = tf.divide(dividend_right, dividor_right)
        # min_right = tf.subtract(1.0, tf.multiply(2.0, tf.divide(dividend_right, dividor_right)))

        min_func = tf.minimum(min_left, min_right)
        # divisor_1 = tf.subtract(b1, a1)
        # dividend_1 = tf.abs(tf.subtract(x, m1))
        # m_1_func = tf.subtract(1.0, tf.multiply(2.0, tf.divide(dividend_1, divisor_1)), name="m_1")
        m_1 = tf.maximum(min_func, 0.0, name = name)

        return m_1

    # FIXME: Membership function is not working as intended.
    #   - Do some more research why does 

    def normLayer_reshaped(self):
        # Reshape the MFs
        self.reshaped_mfs = tf.reshape(self.mf, shape=(self.num_rules, 1))
        # Normalize the MFs
        self.normalizedMFs = tf.divide(self.reshaped_mfs, tf.reduce_sum(self.reshaped_mfs))

    def defconclussions(self):
        self.conclussions = tf.add(self.a_0, tf.reduce_sum(tf.multiply(self.a_y, self.x)), name="outputs")
        # self.conclussions = tf.reduce_sum(self.conclussions, axis=0)

    # def normLayer(self):
    #     # old outputs
    #     collection = tf.get_collection("outputs")
    #     # new outputs (y')
    #     collection2 = tf.get_collection("y_prims")
    #     for i in range(len(self.mf)):
    #         # self.summ_of_mf = tf.add(self.summ_of_mf, self.mf[i])
    #         self.summ_of_mf = tf.reduce_sum(self.mf)
    #     for i in range(len(self.mf)):
    #         self.normalizedMFs[i] = tf.divide(self.mf[i], self.summ_of_mf)
            # out = tf.get_variable(name="output"+str(i), shape=(), dtype=tf.float32, trainable=0)

            # self.outputs[i] = tf.multiply(self.normalizedMFs[i], collection2[i])

        #
        # size_of_arr = len(self.reshaped_nmfs)
        # # self.y_funcs = tf.reshape(collection2, shape=(size_of_arr, self.num_rules))
        # self.reshaped_nmfs = tf.reshape(self.normalizedMFs, shape=(size_of_arr, 1))
        #
        # self.outputs = tf.reduce_sum(tf.multiply(self.reshaped_nmfs, collection2))


    # def outputTensor(self, index, tensor):
    #     tf.add_to_collection("outputs", tensor)
    #     # self.outputs[index] = tf.multiply(self.mf[index], tensor)

    def outputLayer(self, num_rules, num_inputs):
        # self.outputs[i] = tf.multiply(self.normalizedMFs[])
        #TODO: connection between Produkt layer and Output layer is missing....!!!

        # self.summ = tf.reduce_sum(tf.reshape(self.outputs, (-1, num_rules, num_inputs)), axis=1)
        self.summ = tf.reduce_sum(self.outputs)
        self.optimizeMethod()

    def member_func(self, mf_param=None):
        index = 1
        if len(mf_param) > 0:
            for i in mf_param:
                # a = None
                # m = None
                # b = None
                # with tf.variable_scope("variables", reuse=tf.AUTO_REUSE):
                if(index == 1):
                    print("THis is the index which is getting stopped", index, i[0])
                    a = tf.get_variable(name="a"+str(index), dtype=tf.float32,
                                        initializer=tf.constant(i[0]), trainable=0)
                else:
                    print("This is the index which is not getting stopped", index, i[0])
                    a = tf.get_variable(name="a" + str(index), dtype=tf.float32,
                                        initializer=tf.constant(i[0]), trainable=1)
                m = tf.get_variable(name="m"+str(index), dtype=tf.float32, initializer=tf.constant(i[1]), trainable=1)
                #
                if(index == len(mf_param)):
                    b = tf.get_variable(name="b"+str(index), dtype=tf.float32,
                                        initializer=tf.constant(i[2]), trainable=0)
                else:
                    b = tf.get_variable(name="b" + str(index), dtype=tf.float32,
                                        initializer=tf.constant(i[2]), trainable=1)

                tf.clip_by_value(m, clip_value_min=a, clip_value_max=b)

                self.var[index - 1] = a, m, b

                with tf.variable_scope("mfs", reuse=tf.AUTO_REUSE):
                    self.mf[(index - 1)] = self.triangularMF(self.x, a, m, b, ("mf" + str(index)))

                index += 1
            tf.add_to_collection("mf", self.var)

    def doCalculation(self, sess, x):
        return sess.run(self.result, feed_dict={self.x: x})

    def optimizeMethod(self):
        # self.loss = tf.losses.mean_squared_error(self.y, self.summ)

        self.loss = tf.losses.mean_squared_error(self.y, self.result)

        # self.loss = tf.losses.huber_loss(self.y, self.summ)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(self.loss)

    # überlegung wie man das am besten implementiert, sodass die Conclusionsfunction zu implementieren ist
    # def conc_func(self, num_rules):
    #     # for i in range(num_rules):
    #     tf.add_to_collection("y_prims", tf.add(self.a_0, tf.multiply(self.a_y, self.x)))

    def getVariableInitializer(self):
        return tf.global_variables_initializer()

    def save_graph(self, sess, name, step):
        self.saver.save(sess, dir + "/model/" + name, global_step=step)

    def train(self, sess, x, y):
        return sess.run([self.loss, self.optimizer], feed_dict={self.x: x, self.y: y})

    def fourthLayer(self):
        self.outputs = tf.multiply(self.normalizedMFs, self.conclussions)

    def fifthLayer(self):
        self.result = tf.reduce_sum(self.outputs)
