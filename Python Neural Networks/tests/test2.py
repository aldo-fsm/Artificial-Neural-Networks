'''
Created on 10 de jan de 2017

@author: aldo
'''

from matplotlib import pyplot as plt
import numpy as np

from annpy.neural_nets import NeuralNetwork
from annpy.structures import ActivationFuntions
from annpy.training import DataSet


ann = NeuralNetwork()
ann.default_learning_rate = 0.001
ann.add_layer('i', 1)
ann.add_layer('h1', 10)
ann.add_layer('o', 1, af=ActivationFuntions.LINEAR)

ann.input_layers = 'i'
ann.output_layers = 'o'

ann.connect('i', 'h1', 0.5)
ann.connect('h1', 'o', 0.5)

noise = 3

x = np.arange(-5, 5, 0.2)
y = np.vectorize(lambda x : x ** 3)(x)
y_noise = [i + noise * np.random.randn() for i in y]

training_data = DataSet()
validation_data = DataSet()
for i in range(len(x)) :
    if i % 2 == 0:
        training_data.add_training_case(x[i], y_noise[i])
    else :
        validation_data.add_training_case(x[i], y[i])
        
    
print("treinando com {} exemplos".format(len(training_data)))

n, training_errors, validation_errors = ann.train(training_data, 1000, val_set=validation_data)

plt.plot(training_errors)
plt.plot(validation_errors)
plt.show()

ax = plt.gca()
ax.grid(True)
plt.plot(x, [float(ann.output(i)[0]) for i in x], 'r')
plt.plot(x, y_noise, 'g')
plt.plot(x, y, 'b')
plt.show()
