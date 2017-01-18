'''
Created on 16 de jan de 2017

@author: aldo
'''

from matplotlib import pyplot as plt

from annpy.neural_nets import EchoStateNetwork
from annpy.training import DataSet

esn = EchoStateNetwork(1, 5, 1)
esn.learning_rate = 0.05
esn.set_input_hidden_weights(0.1)
esn.set_hidden_hidden_weights(0.1, 0.7)
esn.set_hidden_bias(1)
esn.set_hidden_output_weights(0.1)
 
data = DataSet()
data.add_training_case(1, 2, 3, 2, 3, 4)
data.add_training_case(4, 5, 6, 5, 6, 7)
data.add_training_case(9, 10, 11, 10, 11, 12)
data.add_training_case(15, 16, 17, 16, 17, 18)
 
errors = []
esn.train(data, 1000, error_list=errors)
 
print(esn.output(1))
print(esn.output(2))
print(esn.output(3))
 
plt.plot(errors)
plt.show()

