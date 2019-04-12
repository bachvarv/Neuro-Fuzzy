import fuzzylite as fl
import tensorflow as tf

tf.enable_eager_execution()


def numb(x, input):
    numb = x.fuzzify(input)
    val = float(numb.partition("/")[0])
    return val


engine = fl.Engine(
    name="Triangular",
    description="2 Set MF"
)

x = fl.InputVariable(
    name="x",
    minimum=0.0,
    maximum=10.0,
    terms=[
        fl.Triangle("",0.00, 3.33, 6.66)
    ]
)

engine.inputs = x

val = x.fuzzify(3.365)

print(numb(x, 4.76))

input = tf.Variable(dtype=tf.float64, name="input", initial_value=1.0)

mf1 = tf.multiply(1.0, numb(x, input))

with tf.Session as sess:
    sess.run(tf.global_variables_initializer())
    value = sess.run(mf1, feed_dict={input: 3.365})
    print(value)

