import numpy as np

# x_data = np.random.randn(2000, 3)
# w_real = [0.3, 0.5, 0.1]
# b_real = -0.2
#
# noise = np.random.randn(1, 2000) * 0.1
# y_data = np.matmul(w_real, x_data.T) + b_real + noise
#
# # Generate dataset
# trnData = x_data[:1500, :] # Ersten 1500
# trnLbls = y_data[:, :1500] # Ersten 1500
# chkData = x_data[-501:,:] # letzten 500
# chkLbls = y_data[:, -501:] # letzten 500

# print(x_data[1000], y_data[0][1000])


def genDataSetWithNoise(input, size):
    x_data = np.random.randn(size, input)

    w_real = np.random.randn(input)
    b_real = np.random.randn(1)

    noise = np.random.randn(1, size)

    y_data = np.matmul(x_data, w_real) + b_real + noise

    # conc_data = np.concatenate((x_data, y_data), axis=1)

    return x_data, y_data, w_real, b_real, noise

x_data, y_data, w_real, b_real, noise = genDataSetWithNoise(2, 1000)

# print(w_real, b_real, noise[0][5])

# print(y_data)

a = [[1, 2], [1,3]]
b = [[1],[2]]

c = np.concatenate((a,b), axis=1)

# print(c)