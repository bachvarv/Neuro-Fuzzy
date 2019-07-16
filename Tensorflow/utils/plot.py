import matplotlib.pyplot as plt

from utils.file_reader import readFile

path = "../utils/sinus.out"
path_2 = "../utils/parabola_1000.out"

x_arr, y_arr = readFile(path)

x_arr_par, y_arr_par = readFile(path_2)

plt.figure(figsize=(4, 3))

plt.plot(x_arr, y_arr)
plt.legend(['f(x) = sin(x)'])

plt.savefig('sinus.png')

plt.figure(figsize=(4,3))
plt.plot(x_arr_par, y_arr_par)
plt.legend(['f(x) = x*x'])

plt.savefig('parabola.png')

# plt.show()