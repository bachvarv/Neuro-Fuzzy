import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

def plot_graph(x_arr, y_arr, name, legend):

    plt.plot(x_arr, y_arr)

    plt.legend(legend)
    plt.show()
    plt.savefig(name)



x_arr = np.linspace(0, 40, 1000)

y_arr = [1 if x >= 10 else 0 for x in x_arr]

y_f_middle = fuzz.trimf(x_arr, [10.0, 18.0, 25.0])
y_f_high = np.subtract([1], fuzz.trapmf(x_arr, [20.0, 30.0, 40.0, 50.0]))
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
plot_graph(x_arr, y_f_high, "not_high_temp.png", ['not high temperature'])
