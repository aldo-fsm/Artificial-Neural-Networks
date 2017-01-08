'''
Created on 8 de jan de 2017

@author: aldo
'''

import neural_net.neural_nets as nn

ann1 = nn.NeuralNetwork()
 
ann1.addLayer("l1", 10)
ann1.addLayer("l2", 15)
ann1.addLayer("l3", 3)
ann1.addLayer("l4", 4)

ann1.connect('l1', 'l2', 3)

ann1.input_layers = 'l1','l3'
ann1.output_layers = 'l2','l4'

print(ann1.input_layers)
print(ann1.output_layers)
