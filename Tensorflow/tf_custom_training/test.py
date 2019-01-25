from random import uniform, random

import tensorflow as tf

from ANFIS.anfis import Anfis

a = [[1.0, 3.0, 5.0], [5.0, 7.0, 9.0]]

num_rules = 2

num_inputs = 1

num_conclusions = 2

f = Anfis(a, num_inputs, num_rules, num_conclusions)

for i in a:
    print(i)

with tf.variable_scope("") as scope:
    scope.reuse_variables()
    a1 = tf.get_variable("a1", [])
    m1 = tf.get_variable("m1", [])
    b1 = tf.get_variable("b1", [])

# mfs = tf.reshape(f.normalizedMFs, shape=[])

f.outputTensor(0, tf.multiply(f.x, 2))

f.outputTensor(1, tf.multiply(f.x, 0.5))

f.outputLayer(num_rules, num_inputs)

# with tf.variable_scope("mfs") as scope:
#     scope.reuse_variables()
#     mf1 = tf.get_variable("mf1", [])

with tf.Session() as sess:
    writer = tf.summary.FileWriter("output_test", graph=sess.graph)

    sess.run(f.init)

    # sess.run(f.summ, feed_dict={f.x:[8]})
    # print(sess.run(f.x[0], feed_dict={f.x[0]: 10}))
    #
    # print(sess.run([f.summ], feed_dict={f.x[0]: 1.3}))

    # print(f.doCalculation(sess, [[7.0]]))

    print(sess.run([a1, m1, b1]))

    for _ in range(500):
        # does work but doesn't seem to change the value of the variables
        # it does not work because the loss function doesn't calculate the right value for errors,
        # when a candidate is beyond the mf's range.
        if uniform(0, 1) < 0.5:
            candidate = uniform(5.01, 6.32)
            y = [[1.0 * candidate]]
        else:
            candidate = uniform(9.2, 12.1)
            y = [[2.0 * candidate]]
        x = [[candidate]]

        print("input:", candidate, f.doCalculation(sess, x))

        print("Train Step:", f.train(sess, x, y))

    with tf.variable_scope("") as scope:
        scope.reuse_variables()
        a1 = tf.get_variable("a1")
        m1 = tf.get_variable("m1")
        b1 = tf.get_variable("b1")
        a2 = tf.get_variable("a2")
        m2 = tf.get_variable("m2")
        b2 = tf.get_variable("b2")

    print(sess.run(f.var))

    writer.close()
