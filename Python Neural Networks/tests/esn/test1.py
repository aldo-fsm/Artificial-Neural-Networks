'''
Created on 16 de jan de 2017

@author: aldo
'''


from annpy.neural_nets import EchoStateNetwork

esn = EchoStateNetwork(3, 100, 2)
esn.set_input_hidden_weights(10)
esn.set_hidden_hidden_weights(20, 0.7)
esn.set_hidden_output_weights(5)

print(esn.output(1, 2, 3, 4, 5, 6, 7, 8, 9))
print(esn.output(1, 2, 3))
