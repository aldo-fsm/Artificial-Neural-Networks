'''
Created on 9 de jan de 2017

@author: aldo
'''
from annpy.neural_nets import NeuralNetwork
from annpy.training import DataSet
from matplotlib import pyplot as plt


ann = NeuralNetwork()
ann.default_learning_rate = 10
ann.add_layer('input', 2)
ann.add_layer('hidden', 5)
ann.add_layer('output', 1)

ann.input_layers = 'input'
ann.output_layers = 'output'

ann.connect('input', 'hidden', 5)
ann.connect('hidden', 'output', 5)

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

plt.plot(errors)
plt.show()
