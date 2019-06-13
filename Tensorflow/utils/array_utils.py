from random import sample
import numpy as np

array = [i * 2 for i in range(1, 101)]


batch = sample(range(len(array)), 20)

batch_x = []
batch_y = []

size = 5

for i in range(size):
    batch_x.append(array[i * 20:(i + 1) * 20])
    batch_y.append(array[i * 20:(i + 1) * 20])

# print(len(array))


fixed_arr = np.reshape(batch_x, (100))

print(fixed_arr)

# print(batch_x)

