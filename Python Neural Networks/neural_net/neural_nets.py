'''
Created on 7 de jan de 2017

@author: aldo
'''

from neural_net.neurons import ActivationFuntions, Layer, SynapticWeights
import numpy as np

class NeuralNetwork:
    
    layers = {}
    default_learning_rate = 0.1
    
    def addLayer(self, name, number_units, **kwargs):
        args = {
            'af':ActivationFuntions.SIGMOID,
            'lr':self.default_learning_rate,
            'p' : 1
            }
        args.update(kwargs)
        layer = Layer(number_units, args['af'], args['lr'])
        if args['af'] == ActivationFuntions.SIGMOID :
            layer.p = args['p']
        self.layers[name] = layer
        
    def connect(self, layer1, layer2, weight_randomization=0.1):
        
        if layer1 in self.layers and layer2 in self.layers:
            
            layer1 = self.layers[layer1]
            layer2 = self.layers[layer2]
            
            random_matrix = (1 - 2 * np.random.rand(layer1.size, layer2.size)) * weight_randomization
            weights = SynapticWeights(random_matrix)
            layer1.weights_out[layer2] = weights
            layer2.weights_in[layer1] = weights

        else :
            raise Exception('Camada {0} ou {1} não existe'.format(layer1, layer2))
        
