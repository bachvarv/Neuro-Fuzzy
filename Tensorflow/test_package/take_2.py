import random

import numpy
import tensorflow as tf

x = tf.placeholder(dtype=tf.float32, name="x")
y = tf.placeholder(dtype=tf.float32, name="y")
x_Value = [1.0, 7.0, 3.40, 8.0, 3.0, 6.0, 0.0, 11.0, 2.5, 9.13]
y_Value = [1.0, 49.0, 11.56, 16.0, 9.0, 36.0, 0.0, 22.0, 6.25, 83.3569]

# y_Value = tf.random_uniform(shape=[5], minval=0.0, maxval=1.0, dtype=tf.float32)

#a1 = tf.Variable(trainable=1, name="a1")
#b1 = tf.Variable(trainable=1, name="b1")


#a2 = tf.Variable(trainable=1, name="a2")
#b2 = tf.Variable(trainable=1, name="b2")


with tf.variable_scope("check_side"):
    center1 = tf.Variable(initial_value=random.uniform(1, 3), name="m_1")
    center2 = tf.Variable(initial_value=random.uniform(6, 8), name="m_2")

    left1 = tf.Variable(initial_value=0.0, name="a_1", dtype=tf.float32)
    right1 = tf.Variable(initial_value=4.0, name="c_1", dtype=tf.float32)

    left2 = tf.Variable(initial_value=5.0, name="a_2", dtype=tf.float32)
    right2 = tf.Variable(initial_value=9.0, name="c_2", dtype=tf.float32)

feed_dict = {x: x_Value}

# # a1 <= x <= m1
# act1 = tf.cast(tf.logical_and(tf.greater_equal(x, left1), tf.greater_equal(center1, x)), dtype=tf.float32,
#                name="a1_less_x_less_m1")
# # m2 <= x <= c1
# act2 = tf.cast(tf.logical_and(tf.greater_equal(x, center1), tf.greater_equal(right1, x)), dtype=tf.float32,
#                name="m1_less_x_less_c1")
#
# # a2 <= x <= m2
# act3 = tf.cast(tf.logical_and(tf.greater_equal(x, left2), tf.greater_equal(center2, x)), dtype=tf.float32,
#                name="a2_lesseq_x_lesseq_m2")
#
# # m2 <= x <= c2
# act4 = tf.cast(tf.logical_and(tf.greater_equal(x, center2), tf.greater_equal(right2, x)), dtype=tf.float32,
#                name="m2_lesseq_x_lesseq_c2")

# if a1 <= x <= m1: (x - a1)/ (m1 - a1)
op_1_left = tf.div(tf.subtract(x, left1), tf.subtract(center1, left1), name="left1")

# if m1 <= x <= b1: (b1 - x) / (b1 - m1)
op_1_right = tf.div(tf.subtract(right1, x), tf.subtract(right1, center1), name="right1")

# # if a2 <= x <= m1: (x-a2)/(m2 - a2)
# op_2_left = tf.div(tf.subtract(x, left2), tf.subtract(center2, left2), name="left2")
#
# # if m2 <= x <= b2: (b2 - x) / (b2 - m2)
# op_2_right = tf.div(tf.subtract(right2, x), tf.subtract(right2, center2), name="right2")

# res_op_1 = tf.multiply(act1, op_1_left, name="res1")
#
# res_op_2 = tf.multiply(act2, op_1_right, name="res2")
#
# res_op_3 = tf.multiply(act3, op_2_left, name="res3")
#
# res_op_4 = tf.multiply(act4, op_2_right, name="res4")

# maximum_1 = tf.maximum(res_op_1, res_op_2)
# maximum_2 = tf.maximum(res_op_3, res_op_4)

fuzzy_func = tf.multiply(op_1_left, tf.multiply(x, x))

fuzzy_func1 = tf.multiply(x, 2)

cost1 = tf.reduce_mean(tf.square(fuzzy_func1 - y))

# cost2 = tf.reduce_mean(tf.square(fuzzy_func1 - y))

optimizer1 = tf.train.GradientDescentOptimizer(0.1).minimize(cost1, var_list=[center1, left1, right1])

# opzimizer2 = tf.train.GradientDescentOptimizer(0.1).minimize(cost2, var_list=[center2, left2, right2])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("output", graph=sess.graph)
    sess.run(init)

    m1 = sess.run(center1)
    m2 = sess.run(center2)

    a1 = sess.run(left1)
    c1 = sess.run(right1)

    print(a1, c1)

    a2 = sess.run(left2)
    c2 = sess.run(right2)

    a = [m1, a1, c1]
    # b = [m2[0], a2[0], c2[0]]

    print("A:", a)
    for i in range(len(x_Value)):
        xVal = x_Value[i]
        yVal = y_Value[i]
        if yVal == (xVal*xVal):
            print(xVal, yVal)
            print(sess.run([optimizer1], feed_dict={x: xVal, y: yVal}))
            print(sess.run([center1, left1, right1]))

    writer.close()
