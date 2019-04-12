from random import uniform
import matplotlib.pyplot as plt
import numpy as np

from utils.data_generation import genDataSetWithNoise

import tensorflow as tf

from ANFIS.anfis import Anfis

x_data, y_data, w_real, b_real, noise = genDataSetWithNoise(1, 2000)

trn_dataX = x_data[:1500, :]
trn_dataY = y_data[:, :1500]

chk_dataX = x_data[-501:, :]
chk_dataY = y_data[:, -501:]

# chk_data = np.concatenate((chk_dataX, chk_dataY), axis=1)

print(chk_dataX[0], chk_dataY)

num_inputs = 2

mat = [[3, 2]]

num_sets = 3

num_conclusions = 2

# mat = [[1, 2]]

# f = Anfis(range=[0.0, 10.0], mat=mat, num_sets=num_sets)

f = Anfis(range=[-6.0, 6.0], mat=mat, num_sets=num_sets)

# for i in a:
#     print(i)

# f.mfs()

with f.sess as sess:
    # sess = tf_debug.TensorBoardDebugWrapperSession(sess, "bachvarv:6064")

    # writer = tf.summary.FileWriter("output_test", graph=sess.graph)

    # uninitialized_vars = []
    # for var in f.all_variables():
    #     try:
    #        sess.run(var)
    #     except tf.errors.FailedPreconditionError:
    #         print(var)
    #         uninitialized_vars.append(var)
    #
    # init_new_vars_op =  tf.variables_initializer(uninitialized_vars)
    sess.run(f.getVariableInitializer())
    #
    print(sess.run(f.var))
    # print("Aktivierungswerte:", sess.run(f.rules_arr, feed_dict={f.x: [2.3]}))
    # print("Zugehörigkeitsfunktionen:", sess.run(f.premisses, feed_dict={f.x: [2.3]}))
    # print(f.rules_arr)

    # print("Result of MF1:", sess.run(f.mf[0], feed_dict={f.x: [2.3]}))
    # print("Result of MF2:", sess.run(f.mf[1], feed_dict={f.x: [2.3]}))
    # print("Result of both MFS:", sess.run(f.mf, feed_dict={f.x: [2.3]}))
    # print(f.mf)
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
    x_val = []
    y_val = []
    y_before_trn = []
    y_out = []

    x = 0.5
    z = 0.5
    index = 0

    plt.figure(figsize=(8.5, 5))

    # while (x < 10.0):
    #     x_val.append([x])
    #     y_val.append(x * x)
    #     y_before_trn.append(f.doCalculation(sess, [x]))
    #     x = x + 0.5
    #     z = z + 0.5
    #     index += 1

    for i in range(len(chk_dataX)):
        x_val.append(chk_dataX[i])
        y_val.append(chk_dataY[0][i])
        y_before_trn.append(f.doCalculation(sess, chk_dataX[i]))

    # plt.plot(x_val, y_val, color='blue')

    plt.scatter(x_val, y_val, color='blue')

    # plt.plot(x_val, y_before_trn, color='green', alpha=0.5)

    plt.scatter(x_val, y_before_trn, color='green', alpha=0.5)

    epochs = len(trn_dataX)
    for r in range(2):
        for i in range(epochs):
            # does work but doesn't seem to change the value of the variables
            # it does not work because the loss function doesn't calculate the right value for errors,
            # when a candidate is beyond the mf's range.
            # if uniform(0, 1) <= 0.5:
            candidate = uniform(0.5, 10.0)
            candidate_2 = uniform(0.5, 9.0)
            y = (candidate * candidate)
            # else:
            #     candidate = uniform(6.0, 9.8)
            #     # print("Candidate:", candidate, "; Erwarteter Resultat", y)
            #     y = 0.5 * candidate
            #     print("Candidate:", candidate, "; Erwarteter Resultat", y)
            x = [candidate]
            # cands.append(candidate)

            # print("Candidate:", candidate, "; Erwarteter Resultat:", y)
            #
            # print("MF0:", sess.run(f.mf[0], feed_dict={f.x: x}))
            # print("MF1:", sess.run(f.mf[1], feed_dict={f.x: x}))
            # print("Summ of MF'S:", sess.run(f.mf, feed_dict={f.x: x}))
            # print("NMF0:", sess.run(f.normalizedMFs, feed_dict={f.x: x}))
            # print("NMF1:", sess.run(f.normalizedMFs[1][0], feed_dict={f.x: x}))
            # print("Output1:", sess.run(f.outputs, feed_dict={f.x: x}))
            # print("Conclussions:", sess.run(f.conclussions, feed_dict={f.x: x}))

            print("Kandidat:", trn_dataX[i], ";Erwarteter Resultat:", trn_dataY[0][i], "; das Model ratet:",
                  f.doCalculation(sess, trn_dataX[i]))

            # print("Kandidat:", candidate, ";Erwarteter Resultat:", y, "; das Model ratet:",
            #       f.doCalculation(sess, x))
            # print("TEST2: input:", candidate, sess.run(f.result, feed_dict={f.x: x}))

            print("Fehlerrate für", candidate, ":", f.train(sess, trn_dataX[i], trn_dataY[0][i]))

            # print("Fehlerrate für", candidate, ":", f.train(sess, x, y))
            print("-----------------------------------------------------")

        # print("a_0", sess.run(f.a_0))
        # print("a_y:", sess.run(f.a_y))
        # print("MFs:", sess.run(f.var))
    # with tf.variable_scope("") as scope:
    #     scope.reuse_variables()
    #     a1 = tf.get_variable("a1")
    #     m1 = tf.get_variable("m1")
    #     b1 = tf.get_variable("b1")
    #     a2 = tf.get_variable("a2")
    #     m2 = tf.get_variable("m2")
    #     b2 = tf.get_variable("b2")

    # print(sess.run(a1))
    a_y = sess.run(f.a_y)
    a_0 = sess.run(f.a_0)
    print("MFs:", sess.run(f.var))
    print("a_0", a_0)
    print("a_y:", a_y)

    # my_labels = ['f(x)= x^2', 'f(x)= ' + str(a_0[0][0]) + ' + ' + str(a_y[0][0])
    #              + ' * x and f(x)= ' + str(a_0[1][0]) + ' + ' + str(a_y[1][0]) + ' * x']

    # for i in range(len(chk_dataX)):
    for i in range(len(x_val)):
        # print(f.doCalculation(sess, [x_val[i]]))
        y_out.append(f.doCalculation(sess, x_val[i]))
        # y_out.append(f.doCalculation(sess, chk_dataX[i]))

    # plt.plot(x_val, y_out, color='red', alpha=0.5)

    plt.scatter(chk_dataX, y_out, color='red', alpha=0.5)

    # loss, result = f.testModell(sess, chk_dataY)

    # print(loss, result)

    # save the graph for export
    # f.save_graph(sess, "model", 1000)

    plt.legend(loc='upper left',
               labels=["Erwartete Werte", "Ergebnisse vor dem Training", "Ergebnisse nach dem Training"])
    # plt.savefig('../graphics/firstgraphics/' + str(epochs)+'_epochs.png')

    plt.show()

    # save the model in a file, which can be opened through tensorboard
    # writer.close()
