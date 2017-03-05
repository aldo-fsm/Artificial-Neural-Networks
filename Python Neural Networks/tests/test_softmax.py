'''
Created on 4 de mar de 2017

@author: aldo
'''
from annpy.neural_nets import NeuralNetwork
from annpy.training import DataSet, ErrorFunctions
from annpy.structures import ActivationFuntions

from matplotlib import pyplot as plt
import numpy as np

ann = NeuralNetwork()
ann.default_learning_rate = 0.3
ann.add_layer('i', 1)
ann.add_layer('h', 30)
ann.add_layer('o', 3, af=ActivationFuntions.SOFTMAX)

ann.input_layers = 'i'
ann.output_layers = 'o'

ann.set_error_functions(ErrorFunctions.CROSS_ENTROPY)

ann.connect('i', 'h', 1)
ann.connect('h', 'o', 1)

data = DataSet()
data.add_training_case(1, 1, 0, 0)
data.add_training_case(2, 0, 1, 0)
data.add_training_case(3, 0, 0, 1)

n, e = ann.train(data, 100)

o = ann.output(1)
print(o)
print(np.sum(o))

plt.plot(e)
plt.show()
