import random
from distutils.command.install_egg_info import install_egg_info

import tensorflow as tf

import numpy as np
from tensorflow.contrib import learn

x = tf.placeholder(dtype=tf.float32, name="X")
y = tf.placeholder(dtype=tf.float32, name="Y")

x_Val_sqr = [random.uniform(1.0, 9.0) for _ in range(50)]
y_Val_sqr = [a for a in x_Val_sqr]

# print(x_Val_sqrt)
# print(y_Val_sqrt)

x_Val_double = [random.uniform(9, 14) for _ in range(50)]
y_Val_double = [(2.0 * x) for x in x_Val_double]

# x_Val = [x for x in x_Val_sqrt x_Val_double]
# y_Val = [y for y in y_Val_sqrt or y_Val_double]

length = len(x_Val_double) + len(x_Val_sqr)

x_Val = [None]*length
y_Val = [None]*length
x_Val[::2] = x_Val_sqr
x_Val[1::2] = x_Val_double

y_Val[::2] = y_Val_sqr
y_Val[1::2] = y_Val_double

# feed_dict = {x: x_Val, y: y_Val}

print(x_Val)
print(y_Val)

a1 = tf.Variable(initial_value=0.0, dtype=tf.float32, name="a1")
b1 = tf.Variable(initial_value=7.0, dtype=tf.float32, name="b1")
m1 = tf.Variable(initial_value=random.uniform(1, 5), dtype=tf.float32, name="m1")

a2 = tf.Variable(initial_value=5.0, dtype=tf.float32, name="a2")
b2 = tf.Variable(initial_value=11.0, dtype=tf.float32, name="b2")
m2 = tf.Variable(initial_value=random.uniform(6, 10), dtype=tf.float32, name="m2")







with tf.name_scope("Layer_1"):
    # m_1 = 1 - abs(x - m1) / b1 - a1
    # second membershipfunction max(min((x-a)/(b-a), (c - x)/(c - b)), 0)
    with tf.name_scope("A1"):
        # min_left = (x - a1) / (m1 - a1)
        min_left = tf.divide(tf.subtract(x, a1), tf.subtract(m1, a1))
        # min_right = (b1 - x) / (b1 - m1)
        min_right = tf.divide(tf.subtract(b1, x), tf.subtract(b1, m1))
        min_func = tf.minimum(min_left, min_right)
        # divisor_1 = tf.subtract(b1, a1)
        # dividend_1 = tf.abs(tf.subtract(x, m1))
        # m_1_func = tf.subtract(1.0, tf.multiply(2.0, tf.divide(dividend_1, divisor_1)), name="m_1")
        m_1 = tf.maximum(min_func, 0.0)

    # m_2 = 1 - 2*abs(x-m2) / b1 - a1
    with tf.name_scope("A2"):
        # min_left = (x - a2) / (m2 - a2)
        min_left = tf.divide(tf.subtract(x, a2), tf.subtract(m2, a2))
        # min_right = (b2 - x) / (b2 - m2)
        min_right = tf.divide(tf.subtract(b2, x), tf.subtract(b2, m2))
        min_func = tf.minimum(min_left, min_right)

        # dividend_2 = tf.abs(tf.subtract(x, m2))
        # divisor_2 = tf.subtract(b2, a2)
        #
        # m_2_func = tf.subtract(1.0, tf.multiply(2.0, tf.divide(dividend_2, divisor_2)), name="m_2")
        m_2 = tf.maximum(min_func, 0.0)

with tf.name_scope("Layer_2_3"):
    weightSum = tf.add(m_1, m_2)
    with tf.name_scope("N1"):
        # Kommt zur Division von 0
        w1_norm = tf.divide(m_1, weightSum, name="w_1")
    with tf.name_scope("N2"):
        w2_norm = tf.divide(m_2, weightSum, name="w_2")


with tf.name_scope("Layer_4"):
    with tf.name_scope("w1_f1"):
        w1_f1 = tf.multiply(w1_norm, tf.multiply(2.0, x), name="w1_f1")
    with tf.name_scope("w2_f2"):
        w2_f2 = tf.multiply(w2_norm, tf.sqrt(x), name="w2_f2")

with tf.name_scope("Layer_5"):
    result = tf.add(w1_f1, w2_f2, name="f")
    # result = tf.maximum(result, 0)

# loss = tf.squared_difference(y, result)
loss = tf.losses.huber_loss(y, result)

# loss = tf.squared_difference(y, result)

# optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

init = tf.global_variables_initializer()

i = 2
# Cliping
# clip_a1 = tf.clip
# clip_a1 = tf.clip_by_value(a1, 0.0, m1)
#
# clip_b1 = tf.clip_by_value(b1, m1, m2)
#
# clip_m1 = tf.clip_by_value(m1, a1, b1)
#
# clip_a2 = tf.clip_by_value(a2, 0.0, m2)
#
# clip_b2 = tf.clip_by_value(b2, m2, 100.0)
#
# clip_m2 = tf.clip_by_value(m2, a2, b2)

with tf.Session() as sess:


    writer = tf.summary.FileWriter("output", graph=sess.graph)
    sess.run(init)

    print("Initial Values for a1, b1, m1:", sess.run([a1, b1, m1]))
    print("Initial Values for a2, b2, m2:", sess.run([a2, b2, m2]))

    for i in range(len(y_Val)):
        feed_dict = {x: x_Val[i], y: y_Val[i]}
        _, cost = sess.run([optimizer, loss], feed_dict=feed_dict)
        res = sess.run([result], feed_dict=feed_dict)
        print("X-Value:", x_Val[i], ";Expected Value:", y_Val[i], "; Computed Value:", res, "; Loss:", cost)
        print()

    # while True:
    for _ in range(100):
        # cost_sum = 0
        # for i in range(len(x_Val)):
        _, cost, res, ms1, ms2, par11, par12, cent1, par21, par22, cent2 = \
            sess.run([optimizer, loss, result, m_1, m_2, a1, b1, m1, a2, b2, m2], feed_dict={x: x_Val, y: y_Val})
        # cost_sum += cost
        print("Cost:", cost)
        # print("X_Val:", x_Val, "Result:", res, "Real Value:", y_Val)
        print("m_1:", ms1)
        print("m_2:", ms2)
        print("A1=[", par11, cent1, par12, "] A2=[", par21, cent2, par22, "]")
        print("-------------------------------------------------------------------------------------")
        # print(cost_sum)


        # cost_sum = sess.run(loss, feed_dict={x: x_Val, y: y_Val})
        # if cost_sum < 100:
        #     break
        # sess.run(tf.assign(a1, clip_a1))

    # print(y_Val_sqrt)
    # result_val = sess.run([result], feed_dict=feed_dict)
    # print(sess.run([result], feed_dict=feed_dict))
    # loss_val = sess.run([loss], feed_dict=feed_dict)

    print(sess.run([a1, b1, m1]))
    print(sess.run([a2, b2, m2]))

    print(sess.run([result, loss], feed_dict={x: 10, y: 20}))
    print(sess.run([result, loss], feed_dict={x: 13, y: 26}))

    # print(cost_sum)
    # # for i in range(len(y_Val)):
    #     print("Expected Value:", result_val[i], "; Computed Value:", y_Val[i], "; Error:",
    #           loss_val[i])

    writer.close()

