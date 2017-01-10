'''
Created on 9 de jan de 2017

@author: aldo
'''
from annpy.neural_nets import NeuralNetwork
from annpy.training import DataSet

ann1 = NeuralNetwork()

ann1.add_layer("input", 2, p=1000)
ann1.add_layer("output", 1, lr=1)
ann1.input_layers = 'input'
ann1.output_layers = 'output'
ann1.connect('input', 'output', 2)

data = DataSet()
data.add_training_case(1, 1, 1)
data.add_training_case(1, 0, 1)
data.add_training_case(0, 1, 1)
data.add_training_case(0, 0, 0)

print(ann1.in_weights_of('output'))
ann1.train(data, 10)
print(ann1.in_weights_of('output'))

print(ann1.output(1, 1))
print(ann1.output(1, 0))
print(ann1.output(0, 1))
print(ann1.output(0, 0))

