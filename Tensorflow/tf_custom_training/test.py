import tensorflow as tf

from ANFIS.anfis import Anfis

a = [[1.0, 3.0, 5.0], [5.0, 7.0, 9.0]]

f = Anfis(a, 1)

for i in a:
    print(i)

with tf.variable_scope("variables") as scope:
    scope.reuse_variables()
    a1 = tf.get_variable("a1", [])
    m1 = tf.get_variable("m1", [])
    b1 = tf.get_variable("b1", [])
# with tf.variable_scope("mfs") as scope:
#     scope.reuse_variables()
#     mf1 = tf.get_variable("mf1", [])

with tf.Session() as sess:
    # writer = tf.summary.FileWriter("output_test", graph= sess.graph)

    sess.run(f.init)
    # writer.close()
