'''
Created on 7 de jan de 2017

@author: aldo
'''

import numpy as np

from annpy.training import ErrorFunctions
from annpy.structures import ActivationFuntions, Layer, SynapticWeights
from mpmath import matrix




class NeuralNetwork:
        
    def __init__(self):
        self.default_learning_rate = 0.1
        self.layers = {}
        self.error_funtion = ErrorFunctions.SQUARED_ERROR
        self._input_layers = []
        self._output_layers = []

    
    @property
    def input_layers(self):
        return {k for k, v in self.layers.items() if v in self._input_layers}
        
    @input_layers.setter 
    def input_layers(self, *values):
        try:
            self._input_layers = [self.layers[name] for name in values]
        except KeyError as e:
            raise ValueError("{0} not exists".format(e.args))
        
    
    @property
    def output_layers(self):
        return {k for k, v in self.layers.items() if v in self._output_layers}
        
    @output_layers.setter 
    def output_layers(self, *values):
        try:
            self._output_layers = [self.layers[name] for name in values]
        except KeyError as e:
            raise ValueError("{0} not exists".format(e.args))
        
    def add_layer(self, name, number_units, **kwargs):
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
            raise ValueError('Layer {0} or {1} not exists'.format(layer1, layer2))
    
    def _forward_prop(self):
        self._reset_output()
        outputs = [layer.outputs for layer in self._output_layers]
        return outputs
    def _back_prop(self):
        self._reset_error()
        for i in self._input_layers :
            for layer in i.weights_out:
                if not layer._error_ready:
                    layer.errors 
    def train(self, data_set, epochs, mini_batch_size=-1):
        if mini_batch_size <= 0:
            mini_batch_size = len(data_set)  # full batch
        
        training_matrices = data_set.training_matrices(mini_batch_size)
        
        in_sizes = [layer.size for layer in self._input_layers]
        out_sizes = [layer.size for layer in self._output_layers]
        
        inputs = []
        targets = []
        aux0 = 0
        for matrix in training_matrices:
            aux1 = 0
            inputs.append([])
            targets.append([])
            for size in in_sizes:
                inputs[aux0].append(matrix[aux1:aux1 + size])
                aux1 += size
            for size in out_sizes:
                targets[aux0].append(matrix[aux1:aux1 + size])
                aux1 += size
            aux0 += 1
            
        for _ in range(epochs):
            for mb_index in range(len(training_matrices)):
                aux0 = 0
                for layer in self._input_layers:
                    layer.outputs = inputs[mb_index][aux0]
                    aux0 += 1
                
                outputs = self._forward_prop()
                
                errors = []
                if self.error_funtion == ErrorFunctions.SQUARED_ERROR:
                    for i in range(len(targets[mb_index])):
                        errors.append(outputs[i] - targets[mb_index][i])    
                elif self.error_funtion == ErrorFunctions.CROSS_ENTROPY:
                    raise NotImplementedError()
                
                aux0 = 0
                for layer in self._output_layers:
                    layer.errors = errors[aux0]
                    aux0 += 1
                
                self._back_prop()
                
                for layer in self.layers.values():
                    if layer not in self._input_layers:
                        layer.update_weights()
    def weights_between(self, layer1, layer2):
        layer1 = self.layers[layer1]
        layer2 = self.layers[layer2]
        return layer2.weights_in[layer1].matrix
    def in_weights_of(self, layer):
        layer = self.layers[layer]
        return [weights.matrix for weights in layer.weights_in.values()], layer.bias
    
    def output(self, *inputs):
        aux = 0
        for layer in self._input_layers:
            matrix = np.matrix(inputs[aux:layer.size])
            layer.outputs = matrix.transpose()
            aux += layer.size
        return self._forward_prop()
                    
    def _reset_output(self):
        for layer in self.layers.values():
            if layer not in self._input_layers:
                layer._output_ready = False
                            
    def _reset_error(self):
        for layer in self.layers.values():
            if layer not in self._output_layers:
                layer._error_ready = False
            
            
