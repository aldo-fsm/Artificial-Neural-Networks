'''
Created on 9 de jan de 2017

@author: aldo
'''

from matplotlib import pyplot as plt
import numpy as np

from annpy.neural_nets import NeuralNetwork
from annpy.training import DataSet


ann = NeuralNetwork()
ann.default_learning_rate = 10
ann.add_layer('input', 2)
ann.add_layer('hidden', 5)
ann.add_layer('output', 1)

ann.input_layers = 'input'
ann.output_layers = 'output'

ann.connect('input', 'hidden', 0.5)
ann.connect('hidden', 'output', 0.5)

data = DataSet()
data.add_training_case(1, 1, 0)
data.add_training_case(1, 0, 1)
data.add_training_case(0, 1, 1)
data.add_training_case(0, 0, 0)

errors = []

ann.train(data, 1000, error_list=errors)

print(ann.output(1, 1))
print(ann.output(1, 0))
print(ann.output(0, 1))
print(ann.output(0, 0))


x = np.linspace(-0.5, 1.5, 50)
y = np.linspace(-0.5, 1.5, 50)
 
X, Y = np.meshgrid(x, y)
 
@np.vectorize
def f(x, y):
    return float(ann.output(x, y)[0])
Z = f(X, Y)
plt.pcolormesh(X, Y, Z, cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.scatter([1, 0], [1, 0], marker='o', color='y')
plt.scatter([1, 0], [0, 1], marker='x', color='y')
plt.axis([-0.5, 1.5, -0.5, 1.5])
plt.show()


# plt.plot(errors)
# plt.show()
