from ANFIS.anfis import Anfis
import tensorflow as tf

# epochs = [5, 10, 50, 100]
epochs = [5, 10]

path="../utils/parabola_1001.out"

# for i in range(2, 5):
f = Anfis(range=[-0.5, 200.5], num_sets=3,
              path="../utils/parabola_1001.out", fulltrain=True)

