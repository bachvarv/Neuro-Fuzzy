from itertools import product

import tensorflow as tf
import itertools as it

import numpy.random as npr

# from utils.array_utils import buildTestData

x = tf.constant([1, 2])
y = tf.constant([3, 4])
z = tf.constant([5, 6])

all = [1, 2, 3, 4, 5, 6]

meshgrid = tf.stack(tf.meshgrid(z, tf.meshgrid(x, y, indexing='xy'), indexing='xy'), axis=-1)

# prod = tf.reduce_prod(all, axis=1)

# arr = [[None]*4]*2

arr = [[1, 2], [3, 4], [5, 6]]

items = [None]*(2**3)

# print(arr)

# print(items)

index = 0

# for i in it.product(range(2), repeat=1):
#     # print("Permutation of Items:", i, "; First Item with index", i[0], "-", arr[0][i[0]],
#     #       "; Second Item with index", i[1], "-", arr[1][i[1]], "; Third Item with index", i[2], "-", arr[2][i[2]])
#     # print("Calculating:", arr[0][i[0]], "*", arr[1][i[1]], "*", arr[2][i[2]])
#     print(i)
#     tmp = 1
#     for f in range(len(i)):
#         tmp = arr[f][i[f]] * tmp
#         print("Key f:", f, "; Key i[f]:", i[f])
#         # print(tmp)
#     items[index] = tmp
#     index += 1


# val = 10 / ((num_set * 3) - 1)

ind = 0


test = [[0,1,2],
        [3,4,5]
        ]

# x = test[0:][:-1]

# y = test[0][-1]

# x, y = buildTestData(test)


#
# print(x)
#
# print(y)


num_set = 2

num_input = 1

a = -4

b = 4

c = abs(b - a)

interval = c / ((num_input * num_set * 3) - 1)

print(interval)

y = -4

for i in range(num_input * num_set):
        print(y)
        y = y + interval
        print(y)
        y = y + interval
        print(y)
        y = y + interval


# for i in range(num_input):
#     for t in range(num_set):
#         if( t == 0):
#             print((t + ind) * val)
#         else:
#             print((t + ind - 2) * val)
#         ind += 1
#         print((t + ind) * val)
#         ind += 1
#         if(t != num_set - 1):
#             print((t + ind) * val)
#         else:
#             print(10)
#         print("__________")
#
# # print(val)
#
# num_inputs = 1
# num_sets = 2
# for i in range(num_inputs):
#     for ind in range(num_sets):
#         s = num_inputs * (i+1) + ind
        # print(num_inputs*(i + 1) + f)


# s = [[npr.uniform(0.0, 10.0) for s in range(2)] for k in range(2)]

# print(s)

# print(items)
#         print("Value from the product Item:", i[f], "With the index:", f)

# z = tf.logical_and(x, y)

#
# with tf.Session() as sess:
#     print(sess.run(prod))




