import tensorflow as tf

a = [None]*2
a[0] = [0.4]
a[1] = [0.2]

print(a)
f = tf.reshape(a, shape=(2, 1))


x = tf.placeholder(name="x", shape=(None, 1), dtype=tf.float32)

# left side of the equation

dividend = tf.abs(tf.subtract(x, 1.0))
dividor = tf.subtract(3.0, 1.0)
mfleft_1 = tf.subtract(1.0, tf.multiply(tf.divide(dividend, dividor), 2.0))

# mfleft_1afterSubtracting = tf.subtract(1.0, mfleft_1)

# right side of equation mf
dividend_1 = tf.subtract(5.0, x)
dividor_1 = tf.subtract(5.0, 3.0)
mfright_1 = tf.abs(
    tf.subtract(1.0, tf.multiply(tf.divide(dividend, dividor), 2.0)))

# mfright_1AfterSubtracting = tf.subtract(1.0, mfright_1)

mf_1 = tf.maximum(tf.minimum(mfright_1, mfleft_1), 0.0)
# mf_1 = tf.maximum(tf.minimum(mfleft_1afterSubtracting, mfright_1AfterSubtracting), 0.0)

# testing why something is none

# left side of mf_2

dividend_21 = tf.subtract(x, 5.0)
dividor_21 = tf.subtract(7.0, 5.0)
mfleft_2 = tf.abs(
    tf.subtract(1.0, tf.multiply(tf.divide(dividend_21, dividor_21), 2.0)))

mfleft_2AfterSubtracting = tf.subtract(1.0, mfleft_2)

# top of the division
dividend_22 = tf.subtract(9.0, x)

# dividor

dividor_22 = tf.subtract(9.0, 7.0)

mfright_2 = tf.abs(
    tf.subtract(1.0, tf.multiply(tf.divide(dividend_22, dividor_22), 2.0)))
mfright_2AfterSubtracting = tf.subtract(1.0, mfright_2)

# tf.subtract(1.0,
# mf_2 = tf.maximum(tf.minimum(mfleft_2AfterSubtracting, mfright_2AfterSubtracting), 0.0)
mf_2 = tf.maximum(tf.minimum(mfleft_2, mfright_2), 0.0)


mfsumm = tf.add(mf_1, mf_2)

mfweight1 = tf.divide(mf_1, mfsumm)

mfweight2 = tf.divide(mf_2, mfsumm)

mfresult1 = tf.multiply(mfweight1, tf.multiply(x, 2.0))

mfresult2 = tf.multiply(mfweight2, tf.multiply(x, 0.5))

# mfresult1 = tf.multiply(mf_1, tf.multiply(x, 2.0))
#
# mfresult2 = tf.multiply(mf_2, tf.multiply(x, 0.5))

result = tf.add(mfresult1, mfresult2)

with tf.Session() as sess:
    print(sess.run(f))
    # print(sess.run(x, feed_dict={x: [[3]]}))

    # print("MF1:", sess.run(mf_1, feed_dict={x: [[5.5]]}))

    # print("Result 1:", sess.run(mfresult1, feed_dict={x: [[9.297]]}))

    # print("MF2:", sess.run(mf_2, feed_dict={x: [[5.5]]}))
    # print("Result 2:", sess.run(mfresult2, feed_dict={x: [[9.297]]}))
    #
    # print("Dividend:", sess.run(dividend, feed_dict={x: [[9.297]]}))
    #
    # print("Dividend_1:", sess.run(dividend_1, feed_dict={x: [[9.297]]}))
    #
    # print("MFleft1:", sess.run(mfleft_1, feed_dict={x: [[9.297]]}))
    # print("MFright1:", sess.run(mfright_1, feed_dict={x: [[9.297]]}))
    #
    # print("MF1 min:", sess.run(mf_1, feed_dict={x: [[9.297]]}))
    # print("MFWeigt 1:", sess.run(mfweight1, feed_dict={x: [[9.297]]}))
    #
    # print("--------------------------------------------------------")
    #
    # print("MFleft2:", sess.run(mfleft_2, feed_dict={x: [[9.297]]}))
    # print("MFright2:", sess.run(mfright_2, feed_dict={x: [[9.297]]}))
    #
    # print("Dividend21:", sess.run(dividend_21, feed_dict={x: [[9.297]]}))
    #
    # print("Dividend_22:", sess.run(dividend_22, feed_dict={x: [[9.297]]}))
    #
    # print("MF2 min:", sess.run(mf_2, feed_dict={x:[[9.297]]}))
    # print("MFWeigt 2:", sess.run(mfweight2, feed_dict={x: [[9.297]]}))

    # print("Weight1:", sess.run(mfweight1, feed_dict={x: [[5.5]]}))
    #
    # print("Weight2:", sess.run(mfweight2, feed_dict={x: [[5.5]]}))

    # print(sess.run(result, feed_dict={x: [[9.297]]}))
