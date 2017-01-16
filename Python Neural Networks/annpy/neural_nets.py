'''
Created on 7 de jan de 2017

@author: aldo
'''

import numpy as np

from annpy.training import ErrorFunctions
from annpy.structures import ActivationFuntions, Layer, SynapticWeights, \
    sigmoid_derivative, sigmoid



class EchoStateNetwork:
    
    def __init__(self, number_inputs, number_hidden, number_outputs):
        self.number_inputs = number_inputs
        self.number_hidden = number_hidden
        self.number_outputs = number_outputs
    
        self.hidden_state = np.matrix(np.zeros([number_hidden, 1]))
        
        self.bias_o = np.matrix(np.zeros((number_outputs, 1)))
        self.bias_h = np.matrix(np.zeros((number_hidden, 1)))
        
        self.output_list = []
    
    def set_input_hidden_weights(self, random_amplitude):
        self.ih_weights = np.matrix(np.random.randn(self.number_inputs, self.number_hidden) * random_amplitude)
        
    def set_hidden_hidden_weights(self, random_amplitude, sparseness):
        
        if sparseness < 0 or sparseness > 1:
            raise ValueError('sparseness deve estar entre 0 e 1')
        
        hh_weights = np.random.randn(self.number_hidden, self.number_hidden) * random_amplitude
        number_zeros = int(hh_weights.size * sparseness)
        for _ in range(number_zeros):
            i = np.random.randint(self.number_hidden)
            j = np.random.randint(self.number_hidden)
            while hh_weights[i, j] == 0 :
                if i < self.number_hidden - 1 :
                    i += 1
                elif j < self.number_hidden - 1 :
                    j += 1
                    i = 0
                else:
                    i, j = 0, 0
            hh_weights[i, j] = 0
        self.hh_weights = np.matrix(hh_weights)
        
    def set_hidden_output_weights(self, random_amplitude):
        self.ho_weights = np.matrix(np.random.randn(self.number_hidden, self.number_outputs) * random_amplitude)
    
    def set_hidden_bias(self, random_amplitude):
        self.bias_h = np.matrix(np.random.randn(self.number_hidden, 1) * random_amplitude)
    
    
    def output(self, *inputs):
        
        input_matrices = []            
        outputs = []
        for i in range(int(len(inputs) / self.number_inputs)):
            aux = i * self.number_inputs
            line = inputs[aux:aux + self.number_inputs]
            input_matrices.append(np.matrix(line).transpose())
        for matrix in input_matrices:
            input_sum = self.bias_h + self.ih_weights.transpose() * matrix \
                + self.hh_weights.transpose() * self.hidden_state
             
            self.hidden_state = sigmoid(input_sum, 1)
        
            output = self.bias_o + self.ho_weights.transpose() * self.hidden_state
            outputs.append(output)
        self.output_list.append(outputs)
        return outputs
    def reset(self):
        self.output_list = []
        self.hidden_state = np.matrix(np.zeros([self.number_hidden, 1]))
        
class NeuralNetwork:
        
    def __init__(self):
        self.default_learning_rate = 0.1
        self.layers = {}
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
        
    def set_error_functions(self, *values):
        for i in range(values):
            self._output_layers[i]._error_function = values[i]
    
    @property
    def output_layers(self):
        return {k for k, v in self.layers.items() if v in self._output_layers}
        
    @output_layers.setter 
    def output_layers(self, *values):
        try:
            self._output_layers = [self.layers[name] for name in values]
            for layer in self._output_layers:
                layer._error_function = ErrorFunctions.SQUARED_ERROR
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
            
            random_matrix = np.random.randn(layer1.size, layer2.size) * weight_randomization
            weights = SynapticWeights(random_matrix)
            layer1.weights_out[layer2] = weights
            layer2.weights_in[layer1] = weights

        else :
            raise ValueError('Layer {0} or {1} not exists'.format(layer1, layer2))
    def _input(self, input_matrix):
        aux = 0
        for layer in self._input_layers:
            layer.outputs = input_matrix[aux:aux + layer.size]
            aux += layer.size
    def _output_layer_error_derivative(self, targets_matrix):
        aux = 0
        for layer in self._output_layers:
            target = targets_matrix[aux: aux + layer.size]
            
            if layer._error_function == ErrorFunctions.SQUARED_ERROR:
                errors = layer.outputs - target
            elif layer._error_function == ErrorFunctions.CROSS_ENTROPY:
                raise NotImplementedError()

            if layer.activation_function == ActivationFuntions.SIGMOID:
                errors = np.multiply(sigmoid_derivative(layer.outputs, layer.p), errors)
            elif layer.activation_function == ActivationFuntions.SOFTMAX:
                raise NotImplementedError()
            
            layer.errors = errors
            aux += layer.size
    def _output_layer_error(self, targets_matrix):
        aux = 0
        for layer in self._output_layers:
            target = targets_matrix[aux: aux + layer.size]
            
            if layer._error_function == ErrorFunctions.SQUARED_ERROR:
                errors = np.power(layer.outputs - target, 2) * 0.5
            elif layer._error_function == ErrorFunctions.CROSS_ENTROPY:
                raise NotImplementedError()
            
            return np.sum(errors)
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
    def train(self, training_set, epochs, mini_batch_size=-1, **kwargs):
        if mini_batch_size <= 0:
            mini_batch_size = len(training_set)  # full batch
                
        training_matrices = training_set.training_matrices(mini_batch_size)
        
        number_inputs = sum([layer.size for layer in self._input_layers])
        
        input_matrices = []
        target_matrices = []
        for matrix in training_matrices:
            input_matrices.append(matrix[:number_inputs])
            target_matrices.append(matrix[number_inputs:])
            
        for _ in range(epochs):
            for mb_index in range(len(training_matrices)):

                self._input(input_matrices[mb_index])
                self._forward_prop()
                self._output_layer_error_derivative(target_matrices[mb_index])
                self._back_prop()
                
                if 'error_list' in kwargs :
                    error = self._output_layer_error(target_matrices[mb_index])
                    kwargs['error_list'].append(error)
                
                # atualiza todos os pesos e bias da rede
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
        input_matrix = np.matrix(inputs).transpose()
        self._input(input_matrix)
        return self._forward_prop()
               
    def _reset_output(self):
        for layer in self.layers.values():
            if layer not in self._input_layers:
                layer._output_ready = False
                            
    def _reset_error(self):
        for layer in self.layers.values():
            if layer not in self._output_layers:
                layer._error_ready = False
                
            
            
