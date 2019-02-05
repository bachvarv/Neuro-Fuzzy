import tensorflow as tf

class Anfis:
    # num_inputs is not being used. trying the simple way of calculating
    def __init__(self,num_inputs, num_rules, num_conclusions, mf_param=None):
        size = len(mf_param)
        self.mf = [None]*(size)
        self.summ = []
        self.x = None
        self.normalizedMFs = [None]*(num_rules)
        self.outputs = [None]*(num_conclusions)
        self.optimizer = None
        self.loss = None
        self.var = [None] * size
        self.summ_of_mf = tf.get_variable(name="summOfMF", dtype=tf.float32, initializer=tf.constant(0.0))
        self.y = tf.placeholder(name="y", shape=(1, 1), dtype=tf.float32)
        for i in range(num_inputs):
            self.x = tf.placeholder(name="x"+str(i + 1), shape=(1, 1), dtype=tf.float32)
        self.mfParameters = mf_param
        # print(len(self.mfParameters))
        self.member_func(mf_param)
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

        self.trainableParams = tf.trainable_variables()
        # self.init = tf.global_variables_initializer()

    def triangularMF(self, x, a, m, b, name):
        # min_left = (x - a1) / (m1 - a1)
        # dividend = tf.abs(tf.subtract(x, a))
        dividend = tf.subtract(x, a)
        dividor = tf.subtract(m, a)
        # min_left = tf.abs(tf.divide(dividend, dividor))
        min_left = tf.divide(dividend, dividor)

        # min_right = (b1 - x) / (b1 - m1)
        # dividend_right = tf.abs(tf.subtract(b, x))
        dividend_right = tf.subtract(b, x)
        dividor_right = tf.subtract(b, m)

        # min_right = tf.abs(tf.divide(dividend_right, dividor_right))
        min_right = tf.divide(dividend_right, dividor_right)

        min_func = tf.minimum(min_left, min_right)
        # divisor_1 = tf.subtract(b1, a1)
        # dividend_1 = tf.abs(tf.subtract(x, m1))
        # m_1_func = tf.subtract(1.0, tf.multiply(2.0, tf.divide(dividend_1, divisor_1)), name="m_1")
        m_1 = tf.maximum(min_func, 0.0, name = name)

        return m_1

    # FIXME: Membership function is not working as intended.
    #   - Do some more research why does 

    def normLayer(self):
        for i in range(len(self.mf)):
            self.summ_of_mf = tf.add(self.summ_of_mf, self.mf[i])
        for i in range(len(self.mf)):
            self.normalizedMFs[i] = tf.divide(self.mf[i], self.summ_of_mf)

    def outputTensor(self, index, tensor):
        self.outputs[index] = tf.multiply(self.normalizedMFs[index], tensor)
        # self.outputs[index] = tf.multiply(self.mf[index], tensor)

    def outputLayer(self, num_rules, num_inputs):
        # self.outputs[i] = tf.multiply(self.normalizedMFs[])
        #TODO: connection between Produkt layer and Output layer is missing....!!!

        self.summ = tf.reduce_sum(tf.reshape(self.outputs, (-1, num_rules, num_inputs)), axis=1)

        self.optimizeMethod()

    def member_func(self, mf_param=None):
        index = 1
        if len(mf_param) > 0:
            for i in mf_param:
                # a = None
                # m = None
                # b = None
                # with tf.variable_scope("variables", reuse=tf.AUTO_REUSE):
                a = tf.get_variable(name="a"+str(index), dtype=tf.float32, initializer=tf.constant(i[0]), trainable=1)
                m = tf.get_variable(name="m"+str(index), dtype=tf.float32, initializer=tf.constant(i[1]), trainable=1)
                b = tf.get_variable(name="b"+str(index), dtype=tf.float32, initializer=tf.constant(i[2]), trainable=1)
                self.var[index - 1] = [a, m, b]

                with tf.variable_scope("mfs", reuse=tf.AUTO_REUSE):
                    self.mf[(index - 1)] = self.triangularMF(self.x, a, m, b, ("mf" + str(index)))

                index += 1
            self.normLayer()


    def doCalculation(self, sess, x):
        return sess.run(self.summ, feed_dict={self.x: x})

    def optimizeMethod(self):
        self.loss = tf.losses.mean_squared_error(self.y, self.summ)

        # self.loss = tf.losses.huber_loss(self.y, self.summ)

        # self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.loss)
        # self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.1).minimize(self.loss)

    def getVariableInitializer(self):
        return tf.global_variables_initializer()

    def train(self, sess, x, y):
        return sess.run([self.loss, self.optimizer], feed_dict={self.x: x, self.y: y})
