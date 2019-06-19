from random import uniform
import matplotlib.pyplot as plt
import time
from ANFIS.anfis import Anfis

start = time.process_time()
num_sets = 4


f = Anfis(num_sets=num_sets,
          path="../utils/sinus.out", gradient_type=1)

with f.sess as sess:
    sess.run(f.get_variable_initializer())

    x_val = []
    y_val = []
    y_before_trn = []
    y_out = []

    x = 0.5
    z = 0.5
    index = 0

    candidate = uniform(0.5, 10.0)
    candidate_2 = uniform(0.5, 9.0)
    y = (candidate * candidate)

    x = [candidate]

    print("Training Time %fs." % f.train(sess, 50))
    print("-----------------------------------------------------")

    a_y = sess.run(f.a_y)
    a_0 = sess.run(f.a_0)
    print("a_0", a_0)
    print("a_y:", a_y)

    end = time.process_time()
    print("Time of the program from start to finnish: %fs" % (end-start))
    plt.show()