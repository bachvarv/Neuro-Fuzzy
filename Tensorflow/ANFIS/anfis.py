import tensorflow as tf

class Anfis:
    # num_inputs is not being used. trying the simple way of calculating
    def __init__(self, mfParameters, num_inputs, num_rules, num_conclusions):
        size = len(mfParameters)
        self.mf = [None]*(size)
        self.summ = []
        self.x = None
        self.normalizedMFs = [None]*(num_rules)
        self.outputs = [tf.placeholder(dtype=tf.float32, shape=None)]*(num_conclusions)
        self.optimizer = None
        self.loss = None
        self.var = [None] * size
        self.y = tf.placeholder(name="y", shape=(1, 1), dtype=tf.float32)
        for i in range(num_inputs):
            self.x = tf.placeholder(name="x"+str(i + 1), shape=(1, 1), dtype=tf.float32)
        self.mfParameters = mfParameters
        # print(len(self.mfParameters))
        index = 1
        for i in mfParameters:
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

        # self.result = tf.

        # output layer selbst erzeugen am Ende
        #self.outputLayer(num_rules, num_inputs)

        # self.optimizeMethod()

        self.trainableParams = tf.trainable_variables()
        self.init = tf.global_variables_initializer()

    def triangularMF(self, x, a, m, b, name):
        # min_left = (x - a1) / (m1 - a1)
        min_left = tf.divide(tf.abs(tf.subtract(x, a)), tf.subtract(m, a))
        # min_right = (b1 - x) / (b1 - m1)
        min_right = tf.divide(tf.abs(tf.subtract(b, x)), tf.subtract(b, m))
        min_func = tf.minimum(min_left, min_right)
        # divisor_1 = tf.subtract(b1, a1)
        # dividend_1 = tf.abs(tf.subtract(x, m1))
        # m_1_func = tf.subtract(1.0, tf.multiply(2.0, tf.divide(dividend_1, divisor_1)), name="m_1")
        m_1 = tf.maximum(min_func, 0.0, name= name)

        return m_1

    def normLayer(self):
        summ = 0.0
        for i in range(len(self.mf)):
            summ = tf.add(summ, self.mf[i])
        for i in range(len(self.mf)):
            self.normalizedMFs[i] = tf.divide(self.mf[i], summ)

    def outputTensor(self, index, tensor):
        self.outputs[index] = tf.multiply(self.normalizedMFs[index], tensor)

    def outputLayer(self, num_rules, num_inputs):
        # self.outputs[i] = tf.multiply(self.normalizedMFs[])
        #TODO: connection between Produkt layer and Output layer is missing....!!!

        self.summ = tf.reduce_sum(tf.reshape(self.outputs, (-1, num_rules, num_inputs)), axis=1)

        self.optimizeMethod()

    def doCalculation(self, sess, x):
        return sess.run(self.summ, feed_dict={self.x: x})

    def optimizeMethod(self):
        self.loss = tf.losses.mean_squared_error(self.y, self.summ)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(self.loss)

    def train(self, sess, x, y):
        return sess.run([self.loss, self.optimizer], feed_dict={self.x: x, self.y: y})
