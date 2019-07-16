
import tensorflow as tf

def calc_mf(x):
    dividend = tf.abs(tf.subtract(-59.399368053425, x))
    divider = tf.subtract(-20.6004675644439, -100.0005)
    op = tf.divide(dividend, divider)
    mul = tf.multiply(tf.cast([2.], tf.float32), op)
    sub = tf.subtract(tf.cast([1.], tf.float32), mul)
    max = tf.maximum(tf.cast([0.], tf.float32), sub, name="mf")

    return max

def calc(x):
    return (-39.16341495 + (-62.30142877*x))

def calc1(x):
    return (-811.39065146 + (-104.3525183 * x))
print(calc(-99.7))
# print(calc1(-99))

# x = tf.placeholder(dtype=tf.float32, shape=())
# y = tf.placeholder(dtype=tf.float32, shape=())
# func = tf.losses.mean_squared_error(x, y)

a = tf.cast(-102, tf.float64)
m = tf.cast(-59.399368053425, tf.float64)
b = tf.cast(-20.6004675644439, tf.float64)
x = tf.cast(-99.7997997997998, tf.float64)

dividend = tf.abs(tf.subtract(x, m))
divider = tf.abs(tf.subtract(b, a))
op = tf.divide(dividend, divider)
mul = tf.multiply(tf.cast([2], tf.float64), op)
sub = tf.subtract(tf.cast([1], tf.float64), mul)
max = tf.maximum(tf.cast([0], tf.float64), sub, name="mf")

range = tf.subtract(m, tf.divide(tf.subtract(b, a), 2))
range_right = tf.add(m, tf.divide(tf.subtract(b, a), 2))

# mf = calc_mf(-99.7)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Dividend: %s" % sess.run(dividend))
    print("Divider: %s" % sess.run(divider))
    print("op: %s" % sess.run(op))
    print("mul: %s" % sess.run(mul))
    print("sub: %s" % sess.run(sub))

    print("Max: %s" % sess.run(max))

    print("Range: %s" % sess.run(range))
    print("Range: %s" % sess.run(range_right))
    # print(sess.run(func, feed_dict={x: -99.7,}))