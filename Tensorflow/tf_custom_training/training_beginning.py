import tensorflow as tf
import numpy as np

from first_model_in_use.model2 import y_Value

x = tf.placeholder(name="x", dtype=tf.float32)
y = tf.placeholder(name="y", dtype=tf.float32)

w = tf.Variable(initial_value=3.0, dtype=tf.float32, name="w")

b = tf.Variable(initial_value=1.0, dtype=tf.float32, name="b")

op = tf.add(tf.multiply(w, x), b, name="opt_func")

mu = tf.get_variable("mu", [2], initializer=tf.random_normal_initializer(0, 1))

loss = tf.squared_difference(op, y)
optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(40):
        x_Value = np.random.rand()
        y_Value = x_Value * 5.3 + 0.4
        feed_dict = {x: x_Value, y: y_Value}
        print(sess.run(optimizer, feed_dict=feed_dict))

    print(sess.run([w, b]))
