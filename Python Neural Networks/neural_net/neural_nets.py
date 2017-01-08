'''
Created on 7 de jan de 2017

@author: aldo
'''

from neural_net.neurons import ActivationFuntions, Layer, SynapticWeights
import numpy as np

class NeuralNetwork:
    
    layers = {}
    default_learning_rate = 0.1
    
    @property
    def input_layers(self):
        return {k for k, v in self.layers.items() if v in self._input_layers}
        
    @input_layers.setter 
    def input_layers(self, values):
        try:
            self._input_layers = [self.layers[name] for name in values]
        except KeyError as e:
            raise Exception("{0} not exists".format(e.args))
        
    
    @property
    def output_layers(self):
        return {k for k, v in self.layers.items() if v in self._output_layers}
        
    @output_layers.setter 
    def output_layers(self, values):
        try:
            self._output_layers = [self.layers[name] for name in values]
        except KeyError as e:
            raise Exception("{0} not exists".format(e.args))
        
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
            raise Exception('Layer {0} or {1} not exists'.format(layer1, layer2))
    
    def _forward_prop(self):
        outputs = [layer.outputs for layer in self._output_layers]
        self._reset_visited()
        return outputs
    def _back_prop(self):
        for i in self.input_layers :
            for layer in i.weights_out:
                if not layer._visited:
                    layer.errors 
        self._reset_visited()
    def train(self, data_set, epochs, mini_batch_size=-1):
        if mini_batch_size <= 0:
            mini_batch_size = len(data_set)
        #------------------------------------------------------------------
        # FALTA IMPLEMENTAR
        #------------------------------------------------------------------
            
    def _reset_visited(self):
        for layer in self.layers.values():
            layer._visited = False
            
            
