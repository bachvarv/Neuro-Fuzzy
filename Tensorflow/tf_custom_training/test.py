from random import uniform

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
    # print(sess.run(f.x[0], feed_dict={f.x[0]: 10}))
    #
    # print(sess.run([f.summ], feed_dict={f.x[0]: 1.3}))

    # print(f.doCalculation(sess, [[7.0]]))

    print(sess.run([a1, m1, b1]))

    for _ in range(50):
        # does work but doesn't seem to change the value of the variables
        candidate = uniform(1.01, 6.32)
        x = [[candidate]]
        y = [[2.0 * candidate]]
        print("input:", candidate, f.doCalculation(sess, x))

        print(f.train(sess, x, y))

    with tf.variable_scope("") as scope:
        scope.reuse_variables()
        a1 = tf.get_variable("a1")
        m1 = tf.get_variable("m1")
        b1 = tf.get_variable("b1")

    print(sess.run([a1, m1, b1]))

    writer.close()
