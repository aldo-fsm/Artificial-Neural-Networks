'''
Created on 9 de jan de 2017

@author: aldo
'''

from matplotlib import pyplot as plt
import numpy as np

from annpy.neural_nets import NeuralNetwork
from annpy.training import DataSet

ann1 = NeuralNetwork()

ann1.add_layer("input", 2)
ann1.add_layer("output", 1, lr=1, p=5)
ann1.input_layers = 'input'
ann1.output_layers = 'output'
ann1.connect('input', 'output', 1)

data = DataSet()
data.add_training_case(1, 1, 1)
data.add_training_case(1, 0, 1)
data.add_training_case(0, 1, 1)
data.add_training_case(0, 0, 0)

print(ann1.in_weights_of('output'))
n = ann1.train(data, 10000, acceptable_error=0.01)
print('trained with {} epochs'.format(n))
print(ann1.in_weights_of('output'))

print(ann1.output([1, 1], [1, 0], [0, 1], [0, 0]))
 
x = np.linspace(-0.5, 1.5, 50)
y = np.linspace(-0.5, 1.5, 50)
 
X, Y = np.meshgrid(x, y)
 
@np.vectorize
def f(x, y):
    return float(ann1.output(x, y)[0])
Z = f(X, Y)
plt.pcolormesh(X, Y, Z, cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.scatter(0, 0, marker='o', color='y')
plt.scatter([1, 1, 0], [1, 0, 1], marker='x', color='y')
plt.axis([-0.5, 1.5, -0.5, 1.5])
plt.show()
