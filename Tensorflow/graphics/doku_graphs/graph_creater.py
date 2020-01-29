import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

def plot_graph(x_arr, y_arr, name, legend=None):

    plt.figure(figsize=(4,3))
    plt.plot(x_arr, y_arr)
    plt.legend(legend)
    plt.show()
    plt.savefig(name)



x_arr = np.linspace(0, 30, 1000)
y_f_bigger_1 = [1.0 if a > 1.0 else 0.0 for a in x_arr]
y_f_bigger_1 = np.array(y_f_bigger_1)
# print(x_arr)
# print(y_f_bigger_1)

for i in range(1000):
    print('(%f, %f)' % (x_arr[i], y_f_bigger_1[i]))
# y_f_pos = []
#
# for i in x_arr:
#     y_f_pos.append((i - 100) * (i - 100))
#
# legend = ['f(x) = (x - 100) * (x - 100)']

# plot_graph(x_arr, y_f_pos, name="parabola_positive", legend=legend)
# y_f_triangular = fuzz.trimf(x_arr, [0.0, 15.0, 30.0])
# y_f_trapezoidal = fuzz.trapmf(x_arr, [33.0, 43.0, 53.0, 63.0])
# y_f_glock = fuzz.gaussmf(x_arr, 100.0, 10.0)
# y_arr = [1 if x >= 10 else 0 for x in x_arr]
y_f_greater_1 = fuzz.trapmf(x_arr, [1.0, 20.0, 35.0, 40.0])

# y_f_middle = fuzz.trimf(x_arr, [10.0, 18.0, 25.0])
# y_f_high = np.subtract([1], fuzz.trapmf(x_arr, [20.0, 30.0, 40.0, 50.0]))
# print(x_arr)
# print(y_arr)
# print(y_f_func)

# ind = 0
# for i in x_arr:
#     if i >= 10.0 and i <= 10.2:
#         print(i)
#         print(y_f_middle[ind])
#     ind += 1

# plot_graph(x_arr, y_f_middle, "middle_temp.png", 'middle')
plot_graph(x_arr, y_f_greater_1, "fuzz_logic.png", ['Viel größer als 1'])
plot_graph(x_arr, y_f_bigger_1, "classic_logic.png", ['Viel größer als 1'])

# plot_graph(x_arr, y_f_triangular, "mf_types.png")
# plot_graph(x_arr, y_f_trapezoidal, "mf_types.png")
# plot_graph(x_arr, y_f_glock, "mf_types.png")
