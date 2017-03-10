'''
Created on Mar 10, 2017

@author: guest
'''
from annpy.neural_nets import NeuralNetwork
from annpy.training import DataSet, ErrorFunctions
from annpy.structures import ActivationFuntions

from matplotlib import pyplot as plt
import numpy as np

ann = NeuralNetwork()
ann.default_learning_rate = 0.01
ann.add_layer('i', 3)
ann.add_layer('h', 100)
ann.add_layer('o', 2, af=ActivationFuntions.SOFTMAX)

ann.input_layers = 'i'
ann.output_layers = 'o'

ann.set_error_functions(ErrorFunctions.CROSS_ENTROPY)

ann.connect('i', 'h', 0.1)
ann.connect('h', 'o', 0.1)

data = DataSet()
with open('prime_numbers_dataset.csv') as f:
    data.load(f)
data.randomize()
training_set, validation_set = data.split(500 , 500)

epochs, ts_error, vs_error = ann.train(training_set, 100, val_set=validation_set)
print(epochs)
plt.plot(ts_error)
plt.plot(vs_error)
plt.show()
