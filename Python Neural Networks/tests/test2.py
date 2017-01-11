'''
Created on 10 de jan de 2017

@author: aldo
'''

from matplotlib import pyplot as plt

from annpy.neural_nets import NeuralNetwork
from annpy.structures import ActivationFuntions
from annpy.training import DataSet
from random import random

def function(xi, xf, step, f):
    x, y = [], []
    while(xi <= xf):
        x.append(xi)
        y.append(f(xi))
        xi += step
    return x, y


ann = NeuralNetwork()
ann.default_learning_rate = 0.001
ann.add_layer('i', 1)
ann.add_layer('h1', 100)
ann.add_layer('o', 1, af=ActivationFuntions.LINEAR)

ann.input_layers = 'i'
ann.output_layers = 'o'

ann.connect('i', 'h1', 1)
ann.connect('h1', 'o', 1)

x, y = function(-5, 5, 0.1, lambda x: x ** 2)
y_noise = [i + 1*(2 * random() - 1) for i in y]

training_data = DataSet()
for i in range(len(x)) :
    training_data.add_training_case(x[i], y_noise[i])
print(len(training_data))

ann.train(training_data, 1000)

print(ann.output(3))

ax = plt.gca()
ax.grid(True)

plt.plot(x, [float(ann.output(i)[0]) for i in x], 'r')
plt.plot(x, y_noise, 'g')
plt.plot(x, y, 'b')
plt.show()

