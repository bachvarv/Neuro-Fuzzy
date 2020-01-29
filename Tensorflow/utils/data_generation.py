import numpy as np

from utils.file_reader import createFile



def genDataSetWithNoise(input, size):
    x_data = np.random.randn(size, input)

    w_real = np.random.randn(input)
    b_real = np.random.randn(1)

    noise = np.random.randn(1, size)

    y_data = np.matmul(x_data, w_real) + b_real + noise

    return x_data, y_data, w_real, b_real, noise

# x_data, y_data, w_real, b_real, noise = genDataSetWithNoise(1, 2000)



# print(x_data)
# print(y_data)

# createFile(x_data, y_data)

