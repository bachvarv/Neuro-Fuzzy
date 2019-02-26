from random import uniform, random

from tensorflow.python import debug as tf_debug

import tensorflow as tf

from ANFIS.anfis import Anfis

# lineare funktion y' = a0 + a1*x0

a = [[0.0, 3.0, 5.5], [4.5, 7.0, 10.0]]

num_rules = 2

num_inputs = 1

num_conclusions = 2

f = Anfis(mf_param=a, num_inputs=num_inputs, num_rules=num_rules)

for i in a:
    print(i)

with tf.variable_scope("") as scope:
    scope.reuse_variables()
    a1 = tf.get_variable("a1", [])
    m1 = tf.get_variable("m1", [])
    b1 = tf.get_variable("b1", [])

# mfs = tf.reshape(f.normalizedMFs, shape=[])


# f.outputTensor(0, tf.multiply(f.x, 2.0))
# f.outputTensor(1, tf.divide(f.x, 2.))

# FIXME:
# f.normLayer()
# f.outputLayer(num_rules, num_inputs)

# print(f.mf[0])
#
# print(f.outputs[1])



# initializer = f.getVariableInitializer()

# with tf.variable_scope("mfs") as scope:
#     scope.reuse_variables()
#     mf1 = tf.get_variable("mf1", [])

with tf.Session() as sess:
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "bachvarv:6064")

    # writer = tf.summary.FileWriter("output_test", graph=sess.graph)

    sess.run(f.getVariableInitializer())

    # print("Result of MF1:", sess.run(f.mf[0], feed_dict={f.x: [2.3]}))
    # print("Result of MF2:", sess.run(f.mf[1], feed_dict={f.x: [2.3]}))
    # print("Result of both MFS:", sess.run(f.mf, feed_dict={f.x: [2.3]}))
    #
    # print("Reshaped MFs", sess.run(f.reshaped_mfs, feed_dict={f.x: [2.3]}))
    #
    # print("Normalized MFs", sess.run(f.normalizedMFs, feed_dict={f.x: [2.3]}))
    #
    # print("Conclussions:", sess.run(f.conclussions, feed_dict={f.x: [2.3]}))
    #
    # print("Function", sess.run(f.a_0 + (f.a_y * f.x), feed_dict={f.x: [2.3]}))
    # print("Normalized MFS * Fnc", sess.run(f.outputs, feed_dict={f.x: [2.3]}))
    #
    # print("TEST1: input:", 2.3, f.doCalculation(sess, [2.3]))
    #
    # print("A_0", sess.run(f.a_0))
    # print("A_y", sess.run(f.a_y))
    # print("A_0 + A_y", sess.run(f.a_0 + f.a_y))

    # var_coll = tf.get_collection("outputs")

    # print("Y_params:")
    # print(sess.run(f.a_y))
    # print(sess.run(f.a_0))
    # print("------------------------------------------")
    #
    # var_coll = tf.get_collection("y_prims")
    #
    # print("Collection", var_coll)
    # print("Output:", sess.run(var_coll, feed_dict={f.x: [[3.3]]}))

    # print("Variable 1: ", sess.run(a1))
    # summ = tf.reduce_sum(f.outputs)
    # print("Output:", sess.run(summ, feed_dict={f.x: [[3.3]]}))
    # print("-----------------------------------------------------------------")

    # sess.run(f.summ, feed_dict={f.x:[8]})
    # print(sess.run(f.x[0], feed_dict={f.x[0]: 10}))
    # print("MF1:", sess.run( f.mf[0], feed_dict={f.x: [[3.0]]}))
    # print("MF2:", sess.run( f.mf[1], feed_dict={f.x: [[3.0]]}))
    # print(sess.run([f.summ], feed_dict={f.x[0]: 1.3}))

    # print(f.doCalculation(sess, [[7.0]]))

    # print(sess.run([a1, m1, b1]))
    #
    # print("Membership function: ", sess.run(f.mf, feed_dict={f.x: [[3.3]]}))
    # print("Membership function: ", sess.run(f.mf[1], feed_dict={f.x: [[3.3]]}))
    # print("Summ Of MemberShip Function: ", sess.run(f.summ_of_mf, feed_dict={f.x: [[3.3]]}))
    # print("Outputs", sess.run(f.outputs, feed_dict={f.x: [[3.3]]}))
    # print("Reshaped:", sess.run(f.reshaped_nmfs, feed_dict={f.x: [[3.3]]}))
    # print("Reshaped Outputs:", sess.run(f.y_funcs, feed_dict={f.x: [[3.3]]}))

    for _ in range(10000):
        # does work but doesn't seem to change the value of the variables
        # it does not work because the loss function doesn't calculate the right value for errors,
        # when a candidate is beyond the mf's range.
        # if uniform(0, 1) <= 0.5:
        candidate = uniform(0.5, 9.0)
        y = candidate * candidate
        # else:
        #     candidate = uniform(6.0, 9.8)
        #     # print("Candidate:", candidate, "; Erwarteter Resultat", y)
        #     y = 0.5 * candidate
        #     print("Candidate:", candidate, "; Erwarteter Resultat", y)
        x = [candidate]

        print("-----------------------------------------------------")
        print("Candidate:", candidate, "; Erwarteter Resultat:", y)

        print("MF0:", sess.run(f.mf[0], feed_dict={f.x: x}))
        print("MF1:", sess.run(f.mf[1], feed_dict={f.x: x}))
        print("Summ of MF'S:", sess.run(f.mf, feed_dict={f.x: x}))
        print("NMF0:", sess.run(f.normalizedMFs, feed_dict={f.x: x}))
        # print("NMF1:", sess.run(f.normalizedMFs[1][0], feed_dict={f.x: x}))
        print("Output1:", sess.run(f.outputs, feed_dict={f.x: x}))
        print("Conclussions:", sess.run(f.conclussions, feed_dict={f.x: x}))

        print("TEST1: input:", candidate, f.doCalculation(sess, x))
        print("TEST2: input:", candidate, sess.run(f.result, feed_dict={f.x: x}))

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
    print("Self prims:")
    print(sess.run(f.a_y))
    print(sess.run(f.a_0))
    print("------------------------------------------")
    #
    # f.save_graph(sess, "model", 1000)

    # writer.close()
