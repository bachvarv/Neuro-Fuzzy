
import timeit
from random import uniform
import matplotlib.pyplot as plt
import time
from ANFIS.anfis import Anfis

start = time.process_time()
num_sets = 1

f = Anfis(num_sets=num_sets,
          path="../utils/two_input_equation.out", gradient_type=1, mf_type=0)

with f.sess as sess:
    sess.run(f.get_variable_initializer())

    print("Training Time %fs." % f.train(sess, 10))
    print("-----------------------------------------------------")

    a_y = sess.run(f.a_y)
    a_0 = sess.run(f.a_0)
    print("a_0", a_0)
    print("a_y:", a_y)

    end = time.process_time()
    print("Time of the program from start to finnish: %fs" % (end-start))

    plt.show()