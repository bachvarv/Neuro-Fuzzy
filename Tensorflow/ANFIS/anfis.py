import tensorflow as tf


class Anfis:
    #
    def __init__(self, mfParameters, num_inputs):
        size = len(mfParameters)
        for i in range(num_inputs):
            self.x = tf.placeholder(name="x"+str(i), dtype=tf.float32)
        self.mfParameters = mfParameters
        index = 1
        for i in mfParameters:
            with tf.variable_scope("variables", reuse=tf.AUTO_REUSE):
                a = tf.get_variable(name="a"+str(index), dtype=tf.float32, initializer=tf.constant(i[0]))
                m = tf.get_variable(name="m"+str(index), dtype=tf.float32, initializer=tf.constant(i[1]))
                b = tf.get_variable(name="b"+str(index), dtype=tf.float32, initializer=tf.constant(i[2]))

                self.triangularMF(self.x, a, m, b, index)

            index += 1

        self.trainableParams = tf.trainable_variables()
        self.init = tf.global_variables_initializer()

    def triangularMF(self, x, a, m, b, index):
        # min_left = (x - a1) / (m1 - a1)
        with tf.name_scope("mfs"):
            min_left = tf.divide(tf.subtract(x, a), tf.subtract(m, a))
            # min_right = (b1 - x) / (b1 - m1)
            min_right = tf.divide(tf.subtract(b, x), tf.subtract(b, m))
            min_func = tf.minimum(min_left, min_right)
            # divisor_1 = tf.subtract(b1, a1)
            # dividend_1 = tf.abs(tf.subtract(x, m1))
            # m_1_func = tf.subtract(1.0, tf.multiply(2.0, tf.divide(dividend_1, divisor_1)), name="m_1")
            m_1 = tf.maximum(min_func, 0.0, name="mf" + str(index))

        return m_1
