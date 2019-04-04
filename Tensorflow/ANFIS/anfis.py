import tensorflow as tf
import itertools as it



class Anfis:
    # num_inputs is not being used. trying the simple way of calculating
    def __init__(self, num_sets, range, mat=None, num_inputs=None):
        self.num_sets = num_sets
        self.test_possible = False
        # Last index of an array is the expected value.
        # mat array is an array wich contains all the test data.
        if (mat != None):
            self.num_inputs = len(mat[0])
            # self.xArr, self.yArr = buildTestData(mat)
            self.test_possible = True
        elif (num_inputs != None):
            self.num_inputs = num_inputs
        else:
            self.num_inputs = 1

        self.num_rules = num_sets ** self.num_inputs
        self.range = range

        self.sess = tf.Session()

        # Variables to Create the Membership functions
        # self.premisses = [[None] * self.num_sets]*self.num_inputs
        # self.rules_arr = [None] * self.num_rules

        self.a_0 = tf.get_variable(name="a_0", dtype=tf.float32,
                                   initializer=tf.ones(shape=(self.num_rules, 1)), trainable=1)
        self.a_y = tf.get_variable(name="a_y", dtype=tf.float32,
                                   initializer=tf.ones(shape=(self.num_rules, self.num_inputs)), trainable=1)

        self.sess.run(self.a_0.initializer)
        self.sess.run(self.a_y.initializer)
        #
        self.var = [[None] * num_sets] * self.num_inputs

        # Saver to export graph(model)
        self.saver = tf.train.Saver()

        # input variable
        self.x = tf.placeholder(name="x", shape=[self.num_inputs], dtype=tf.float32)
        tf.add_to_collection("xVar", self.x)
        # expected result
        self.y = tf.placeholder(name="y", shape=(), dtype=tf.float32)
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
        # dividend = tf.subtract(x, a)
        # dividor = tf.subtract(m, a)
        dividend = tf.subtract(x, par[0])
        dividor = tf.subtract(par[1], par[0])
        min_left = tf.divide(dividend, dividor)

        # min_right = (b1 - x) / (b1 - m1)
        # dividend_right = tf.subtract(b, x)
        # dividor_right = tf.subtract(b, m)
        dividend_right = tf.subtract(par[2], x)
        dividor_right = tf.subtract(par[2], par[1])
        min_right = tf.divide(dividend_right, dividor_right)

        min_func = tf.minimum(min_left, min_right)
        m_1 = tf.maximum(min_func, 0.0, name=name)

        return m_1

    def normLayer_reshaped(self):
        # Reshape the MFs
        self.reshaped_mfs = tf.reshape(self.mf, shape=(self.num_rules, 1))
        # Normalize the MFs
        self.normalizedMFs = tf.divide(self.reshaped_mfs, tf.reduce_sum(self.reshaped_mfs))

    def defconclussions(self):
        self.conclussions = tf.add(self.a_0, tf.multiply(self.a_y, self.x), name="outputs")

    def outputLayer(self):
        self.summ = tf.reduce_sum(self.outputs)
        self.optimizeMethod()

    # def member_func(self, mf_param=None):
    #     index = 1
    #     if len(mf_param) > 0:
    #         for i in mf_param:
    #             if(index == 1):
    #                 a = tf.get_variable(name="a"+str(index), dtype=tf.float32,
    #                                     initializer=tf.constant(i[0]), trainable=0)
    #             else:
    #                 a = tf.get_variable(name="a" + str(index), dtype=tf.float32,
    #                                     initializer=tf.constant(i[0]), trainable=1)
    #             m = tf.get_variable(name="m"+str(index), dtype=tf.float32, initializer=tf.constant(i[1]), trainable=1)
    #
    #             if(index == len(mf_param)):
    #                 b = tf.get_variable(name="b"+str(index), dtype=tf.float32,
    #                                     initializer=tf.constant(i[2]), trainable=0)
    #             else:
    #                 b = tf.get_variable(name="b" + str(index), dtype=tf.float32,
    #                                     initializer=tf.constant(i[2]), trainable=1)
    #             self.var[index - 1] = a, m, b
    #
    #             with tf.variable_scope("mfs", reuse=tf.AUTO_REUSE):
    #                 self.mf[(index - 1)] = self.triangularMF(self.x, a, m, b, ("mf" + str(index)))
    #
    #             index += 1
    #         tf.add_to_collection("mf", self.var)

    def doCalculation(self, sess, x):
        return sess.run(self.result, feed_dict={self.x: x})

    def optimizeMethod(self):
        self.loss = tf.losses.mean_squared_error(self.y, self.result)
        # self.loss = tf.losses.huber_loss(self.y, self.result)

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

    # def plot(self):

    def mfs(self):
        num_mfs = self.num_sets * self.num_inputs
        premisses = [[None] * self.num_sets] * self.num_inputs

        m = [[None] * self.num_sets] * self.num_inputs

        val_inc = abs((self.range[1] - self.range[0])) / ((self.num_sets * 3) - 1)

        print(abs((self.range[1] - self.range[0])))

        ind = 0

        for i in range(self.num_inputs):

            ind = 0
            val = self.range[0]
            for f in range(self.num_sets):
                if f == 0:
                    a = tf.get_variable(initializer=tf.constant(self.range[0] * 1.0), name="a" + str(i) + str(f),
                                        dtype=tf.float32, trainable=0)
                else:
                    a = tf.get_variable(initializer=tf.constant(val - 2 * val_inc),
                                        name="a" + str(i) + str(f),
                                        dtype=tf.float32)

                val = val + val_inc
                ind += 1

                # if(f == 0):
                #     m = tf.get_variable(name="m" + str(i) + str(f), dtype=tf.float32, trainable=1,
                #                         initializer=tf.constant(2 * val_inc))
                # else:
                m = tf.get_variable(name="m" + str(i) + str(f), dtype=tf.float32, trainable=1,
                                    initializer=tf.constant(val))

                val = val + val_inc

                ind += 1

                if (self.num_sets * i + f) == (num_mfs - 1):
                    b = tf.get_variable(name="b" + str(i) + str(f), dtype=tf.float32,
                                        initializer=tf.constant(self.range[1] * 1.0), trainable=0)
                else:
                    b = tf.get_variable(name="b" + str(i) + str(f), dtype=tf.float32,
                                        trainable=1,
                                        initializer=tf.constant(val))

                val = val + val_inc

                self.var[i][f] = a, m, b

                premisses[i][f] = self.triangularMF(self.x[i], self.var[i][f], "test_mf" + str(i) + str(f))

        index = 0
        for perm in it.product(range(self.num_sets), repeat=(self.num_inputs)):
            tmp = tf.ones(shape=(), dtype=tf.float32)
            for i in range(len(perm)):
                tmp = tf.multiply(premisses[i][perm[i]], tmp)
            self.mf[index] = tmp
            index += 1

    def all_variables(self):
        return tf.global_variables()

    def testModell(self, sess, x_data, y_data):

            return sess.run([self.loss, self.result], feed_dict={self.x: x_data, self.y: y_data})