
import numpy as np
import matplotlib.pyplot as plt


# Sinus
from utils.file_reader import createFile

# z1 = np.linspace(-np.pi, np.pi, 400)
# y = np.sin(z1)
#
# z1.resize(len(z1), 1)
# y.resize(1, len(y))
# ma = np.concatenate((z1,y), axis=0)

# np.savetxt('sinus_values.out', ma,  delimiter = ',')
# ma_in = np.loadtxt('sinus_values.out',  delimiter = ',')

# print(ma_in)
# plt.plot(ma_in[:, 1])
# plt.show()

# print(z1)
# print(y)

# createFile(z1, y, "sinus.out")

# Parabola
z1 = np.linspace(-100, 100, 1000)
y = (z1)*(z1)
#
z1.resize(len(z1), 1)
y.resize(1, len(y))
createFile(z1, y, "parabola_1000.out")
# ma = np.concatenate((z1,y), axis=0)
#
# np.savetxt('parabola_values.out', ma,  delimiter = ',')
# ma_in = np.loadtxt('parabola_values.out',  delimiter = ',')
#
# print(ma_in)
# plt.plot(ma_in[1,:])
# plt.show()

# createFile(z1, y, "parabola_positive.out")

