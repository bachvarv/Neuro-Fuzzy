from random import uniform, random

import tensorflow as tf

from ANFIS.anfis import Anfis

a = [[1.0, 3.0, 5.0], [5.0, 7.0, 9.0]]

num_rules = 2

num_inputs = 1

num_conclusions = 2

f = Anfis(mf_param=a, num_inputs=num_inputs, num_rules=num_rules, num_conclusions=num_conclusions)

for i in a:
    print(i)

with tf.variable_scope("") as scope:
    scope.reuse_variables()
    a1 = tf.get_variable("a1", [])
    m1 = tf.get_variable("m1", [])
    b1 = tf.get_variable("b1", [])

# mfs = tf.reshape(f.normalizedMFs, shape=[])

f.outputTensor(0, tf.multiply(f.x[0], 2))

f.outputTensor(1, tf.multiply(f.x[0], 0.5))

f.outputLayer(num_rules, num_inputs)

print(f.mf[0])

print(f.outputs[1])



# initializer = f.getVariableInitializer()

# with tf.variable_scope("mfs") as scope:
#     scope.reuse_variables()
#     mf1 = tf.get_variable("mf1", [])

with tf.Session() as sess:
    writer = tf.summary.FileWriter("output_test", graph=sess.graph)

    sess.run(f.getVariableInitializer())

    # var_coll = tf.get_collection("outputs")

    # print("Collection", var_coll)
    # print("Trying to get MF1:", sess.run(var_coll[0]))

    print("Variable 1: ", sess.run(a1))

    # sess.run(f.summ, feed_dict={f.x:[8]})
    # print(sess.run(f.x[0], feed_dict={f.x[0]: 10}))
    # print(sess.run(f.mf[0], feed_dict={f.x: [[7.0]]}))
    # print(sess.run([f.summ], feed_dict={f.x[0]: 1.3}))

    # print(f.doCalculation(sess, [[7.0]]))

    print(sess.run([a1, m1, b1]))

    print("Membership function: ", sess.run(f.mf, feed_dict={f.x: [[2.297]]}))
    print("Membership function: ", sess.run(f.mf[1], feed_dict={f.x: [[2.297]]}))
    print("Summ Of MemberShip Function: ", sess.run(f.summ_of_mf, feed_dict={f.x: [[2.297]]}))

    for _ in range(100):
        # does work but doesn't seem to change the value of the variables
        # it does not work because the loss function doesn't calculate the right value for errors,
        # when a candidate is beyond the mf's range.
        if uniform(0, 1) < 0.5:
            candidate = uniform(1.0, 6.32)
            y = [[2.0 * candidate]]
        else:
            candidate = uniform(6.0, 9.8)
            y = [[0.5 * candidate]]
        x = [[candidate]]

        print("-----------------------------------------------------")
        print("Candidate:", candidate, "; Erwarteter Resultat:", y[0])

        print(sess.run(f.normalizedMFs[0][0], feed_dict={f.x: x}))
        print(sess.run(f.normalizedMFs[1][0], feed_dict={f.x: x}))
        print(sess.run(f.outputs[1], feed_dict={f.x: x}))

        print("input:", candidate, f.doCalculation(sess, x))

        print("Train Step:", f.train(sess, x, y))
        print("-----------------------------------------------------")
    with tf.variable_scope("") as scope:
        scope.reuse_variables()
        a1 = tf.get_variable("a1")
        m1 = tf.get_variable("m1")
        b1 = tf.get_variable("b1")
        a2 = tf.get_variable("a2")
        m2 = tf.get_variable("m2")
        b2 = tf.get_variable("b2")

    print(sess.run(a1))

    print(sess.run(f.var))

    f.save_graph(sess, "model", 1000)

    writer.close()
