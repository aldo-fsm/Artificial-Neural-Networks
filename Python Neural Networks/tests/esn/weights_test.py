'''
Created on 16 de jan de 2017

@author: aldo
'''
from annpy.neural_nets import EchoStateNetwork

esn = EchoStateNetwork(3, 10, 2)
esn.set_input_hidden_weights(10)
esn.set_hidden_hidden_weights(20, 0.7)
esn.set_hidden_output_weights(5)

print(esn.ih_weights)
print('-------------------------------')
print(esn.hh_weights)
print('-------------------------------')
print(esn.ho_weights)

