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
        
        self.hidden_state_list = []
        self.output_list = []
        self.learning_rate = 0.1
    
    def set_input_hidden_weights(self, variance):
        self.ih_weights = np.matrix(np.random.randn(self.number_inputs, self.number_hidden) * variance ** 0.5)
        
    def set_hidden_hidden_weights(self, variance, sparseness):
        
        if sparseness < 0 or sparseness > 1:
            raise ValueError('sparseness deve estar entre 0 e 1')
        
        hh_weights = np.matrix(np.random.randn(self.number_hidden, self.number_hidden) * variance ** 0.5)
        number_zeros = int(hh_weights.size * sparseness)
        for _ in range(number_zeros):
            i = np.random.randint(self.number_hidden)
            j = np.random.randint(self.number_hidden)
            aux = 0
            while hh_weights[i, j] == 0 and aux < hh_weights.size:
                if i < self.number_hidden - 1 :
                    i += 1
                elif j < self.number_hidden - 1 :
                    j += 1
                    i = 0
                else:
                    i, j = 0, 0
                aux += 1
            hh_weights[i, j] = 0
        self.hh_weights = np.matrix(hh_weights)
        
    def set_hidden_output_weights(self, variance):
        self.ho_weights = np.matrix(np.random.randn(self.number_hidden, self.number_outputs) * variance ** 0.5)
    
    def set_hidden_bias(self, variance):
        self.bias_h = np.matrix(np.random.randn(self.number_hidden, 1) * variance ** 0.5)
    
    def fprop(self):
        
        outputs = []
        input_matrices = []
        
        for i in range(int(len(self.input) / self.number_inputs)):
            aux = i * self.number_inputs
            matrix = self.input[aux:aux + self.number_inputs]
            input_matrices.append(self.input[aux:aux + self.number_inputs])
        for matrix in input_matrices:
            if len(self.hidden_state_list) == 0:
                self.hidden_state = np.matrix(np.zeros([self.number_hidden, matrix.shape[1]]))
            input_sum = self.bias_h + self.ih_weights.transpose() * matrix \
                + self.hh_weights.transpose() * self.hidden_state 
            self.hidden_state = sigmoid(input_sum, 1)
            self.hidden_state_list.append(self.hidden_state)
            output = self.bias_o + self.ho_weights.transpose() * self.hidden_state
            self.output_list.append(output)
            outputs.append(output)
            
        return outputs
    def output(self, *inputs):
        
        self.input = np.matrix(inputs).transpose()
        return self.fprop()
        
    def reset(self):
        self.output_list = []
        self.hidden_state_list = []
    def train(self, training_set, epochs, mini_batch_size=-1, **kwargs):
        if mini_batch_size <= 0:
            mini_batch_size = len(training_set)  # full batch
                
        training_matrices = training_set.training_matrices(mini_batch_size)
        
        input_matrices = []
        target_matrices = []
        
        for matrix in training_matrices:
            number_inputs = int(len(matrix) * self.number_inputs / (self.number_inputs + self.number_outputs))
            
            input_matrices.append(matrix[:number_inputs])
            target_matrices.append(matrix[number_inputs:])
            
        for _ in range(epochs):
            for mb_index in range(len(training_matrices)):
                self.reset()
                
                self.input = input_matrices[mb_index]
                self.fprop()
                
                aux = 0
                errors = []
                for output in self.output_list:
                    errors.append(output - target_matrices[mb_index][aux:aux + self.number_outputs])
                    aux += self.number_outputs
                aux = 0
                gradients = []
                for error in errors:
                    gradients.append(self.hidden_state_list[aux] * error.transpose()) 
                    aux += 1
                self.ho_weights = self.ho_weights - self.learning_rate * sum(gradients)
                
                self.bias_o = self.bias_o - self.learning_rate * (sum(errors).sum(axis=1))
                
                if 'error_list' in kwargs :
                    aux = 0
                    errors = []                
                    for output in self.output_list:
                        errors.append(0.5 * np.power(output - target_matrices[mb_index][aux:aux + self.number_outputs], 2))
                        aux += self.number_outputs
                    kwargs['error_list'].append(np.sum(sum(errors)))
        self.reset()
    
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
        
    def connect(self, layer1, layer2, variance=1):
        
        if layer1 in self.layers and layer2 in self.layers:
            
            layer1 = self.layers[layer1]
            layer2 = self.layers[layer2]
            
            random_matrix = np.random.randn(layer1.size, layer2.size) * variance ** 0.5
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
        
        has_val_set = 'val_set' in kwargs
        if has_val_set :
            validation_matrix = kwargs['val_set'].training_matrices(len(kwargs['val_set']))[0]   
        training_matrices = training_set.training_matrices(mini_batch_size)
        
        number_inputs = sum([layer.size for layer in self._input_layers])
        
        input_matrices = []
        target_matrices = []
        for matrix in training_matrices:
            input_matrices.append(matrix[:number_inputs])
            target_matrices.append(matrix[number_inputs:])        
        
        if has_val_set:
            validation_input_matrix = validation_matrix[:number_inputs]
            validation_target_matrix = validation_matrix[number_inputs:]
            
        vs_error_list = []
        ts_error_list = []
        
        number_epochs = 0
        for _ in range(epochs):
            print('\nepoch {}'.format(number_epochs + 1))
            training_set_error = 0
            for mb_index in range(len(training_matrices)):

                self._input(input_matrices[mb_index])
                self._forward_prop()
                self._output_layer_error_derivative(target_matrices[mb_index])
                self._back_prop()
                
                training_set_error += self._output_layer_error(target_matrices[mb_index])
                
                # atualiza todos os pesos e bias da rede
                for layer in self.layers.values():
                    if layer not in self._input_layers:
                        layer.update_weights()
            if has_val_set :
                    self._input(validation_input_matrix)
                    self._forward_prop()        
                    validation_set_error = self._output_layer_error(validation_target_matrix)
            number_epochs += 1
            
            training_set_error /= len(training_set)
            ts_error_list.append(training_set_error)
            print('Training set error : {}'.format(training_set_error))
            if has_val_set :
                validation_set_error /= len(kwargs['val_set'])
                vs_error_list.append(validation_set_error)
                print('Validation set error : {}'.format(validation_set_error))
            
            if 'acceptable_error' in kwargs :
                if training_set_error <= kwargs['acceptable_error'] :
                    break
        return_list = [number_epochs, ts_error_list]
        if has_val_set:
            return_list.append(vs_error_list)
        return tuple(return_list)   

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
                
            
            