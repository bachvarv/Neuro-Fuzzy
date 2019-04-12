import skfuzzy as fuzz
import numpy as np
import tensorflow as tf

def calculateMFs(x):
    return fuzz.trimf(x, [0.00, 3.33, 6.66])


x = np.array([3.35])

mfs = fuzz.trimf(x, [0.00, 3.33, 6.66])

xVar = tf.get_variable(name="x", dtype=tf.float64, shape=())

input = tf.get_variable(name="input", dtype=tf.float64, initializer=tf.constant(calculateMFs(xVar)))

align = tf.assign(xVar, x)

with tf.Session() as sess:
    sess.run(align)
    sess.run(tf.global_variables_initializer())

    # print(sess.run(input, feed_dict={xVar: x}))

