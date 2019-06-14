
import timeit
from random import uniform
import matplotlib.pyplot as plt
import time


from utils.data_generation import genDataSetWithNoise

import tensorflow as tf

from ANFIS.anfis import Anfis


# start = timeit.timeit()
start = time.process_time()
x_data, y_data, w_real, b_real, noise = genDataSetWithNoise(1, 2000)

trn_dataX = x_data[:1500, :]
trn_dataY = y_data[:, :1500]

chk_dataX = x_data[-501:, :]
chk_dataY = y_data[:, -501:]

# chk_data = np.concatenate((chk_dataX, chk_dataY), axis=1)

num_inputs = 2

mat = [[3]]

num_sets = 4

num_conclusions = 2

# mat = [[1, 2]]

# f = Anfis(range=[0.0, 10.0], mat=mat, num_sets=num_sets)

# f = Anfis(range=[-3.5, 3.5], num_sets=num_sets,
#           path="../utils/sinus.out", fulltrain=True)

f = Anfis(num_sets=num_sets,
          path="../utils/parabola_1000.out", full_train=True, gradient_type=0)

# for i in a:
#     print(i)

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


    # init_new_vars_op =  tf.variables_initializer(uninitialized_vars)
    sess.run(f.get_variable_initializer())
    # arr_x, arr_y = f.pick_batch()
    # sess.run(f.)
    # print(arr_x)
    # print(arr_y)
    # print(sess.run(f.reshaped_mfs, feed_dict={f.x: arr_x}))
    #
    # mat = tf.reshape(f.reshaped_mfs, shape=(5, 4))
    # print(sess.run(mat, feed_dict={f.x: arr_x}))
    #
    # matrix = tf.reduce_sum(f.reshaped_mfs, 1, keep_dims=True)
    # print(sess.run(matrix, feed_dict={f.x: arr_x}))
    #
    # print(sess.run(f.normalizedMFs, feed_dict={f.x: arr_x}))

    # print(sess.run(f.conclusions, feed_dict={f.x: arr_x}))
    # f.do_calculation(sess, arr_x[0])
    # print(sess.run(f.var))
    #
    # print(sess.run(f.var))
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

    # plt.figure(figsize=(8.5, 6))

    # print(f.plotParam(sess, 223, "MFs before training"))

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

    # print("Kandidat:", trn_dataX[i], ";Erwarteter Resultat:", trn_dataY[0][i], "; das Model ratet:",
    #       f.doCalculation(sess, trn_dataX[i]))

    # print("Kandidat:", candidate, ";Erwarteter Resultat:", y, "; das Model ratet:",
    #       f.doCalculation(sess, x))
    # TRAIN
    # epochs = [1, 9, 40, 50]
    # for i in epochs:
    #     print("Training Time %fs." % f.train(sess, i))

    print("Training Time %fs." % f.train(sess, 20))
    # print("Fehlerrate für", candidate, ":", f.train_2(sess, trn_dataX[i], trn_dataY[0][i]))

    # print("Fehlerrate für", candidate, ":", f.train_2(sess, x, y))
    print("-----------------------------------------------------")

    # save the graph for export
    # f.save_graph(sess, "model", 1000)

    # plt.legend(loc='upper left',
    #            labels=["Erwartete Werte", "Ergebnisse vor dem Training", "Ergebnisse nach dem Training"])
    # plt.savefig('../graphics/firstgraphics/' + str(epochs)+'_epochs.png')

    a_y = sess.run(f.a_y)
    a_0 = sess.run(f.a_0)
    # print("MFs:", f.plotParam(sess, 224, "MFs after Training"))
    # print("a_0", a_0)
    # print("a_y:", a_y)

    end = time.process_time()
    print("Time of the program from start to finnish: %fs" % (end-start))
    plt.show()

    # save the model in a file, which can be opened through tensorboard
    # writer.close()
