'''
Created on 8 de jan de 2017

@author: aldo
'''

import annpy.neural_nets as nn

ann1 = nn.NeuralNetwork()
 
ann1.add_layer("l1", 10)
ann1.add_layer("l2", 15)
ann1.add_layer("l3", 3)
ann1.add_layer("l4", 4)

ann1.connect('l1', 'l2', 3)

ann1.input_layers = 'l1','l3'
ann1.output_layers = 'l2','l4'

print(ann1.input_layers)
print(ann1.output_layers)
