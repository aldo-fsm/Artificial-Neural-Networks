'''
Created on 7 de jan de 2017

@author: aldo
'''

import math
from enum import Enum
import numpy as np
from numpy import matlib


# funções de ativação
class ActivationFuntions(Enum):
    SIGMOID = 0
    SOFTMAX = 1
    LINEAR = 2

# peso sinaptico
class SynapticWeights:
    def __init__(self, weight_matrix):
        self.matrix = weight_matrix

# função sigmoide
@np.vectorize        
def sigmoid(x, a=1):
    try:
        return  1 / (1 + math.exp(-a * x))
    except OverflowError:
        return 0
# derivada da função sigmoide
@np.vectorize        
def sigmoid_derivative(y, a=1):
    return a * y * (1 - y)

class Layer:

    
    def __init__(self, number_units, activation_function=ActivationFuntions.SIGMOID, learning_rate=0.1, **kwargs):
        args = {'p':1}
        args.update(kwargs)
        
        if type == ActivationFuntions.SIGMOID :
            # parametro para a função sigmoide
            self.p = args['p']
            
        # função de ativação dos neuronios da camada
        self.activation_function = activation_function
        # numero de neuronios da camada
        self.size = number_units
        
        self.bias = np.matrix([0 for _ in range(self.size) ]).transpose()
        self.learning_rate = learning_rate
        
        # pesos das entradas
        self.weights_in = {}
        # pesos das saidas
        self.weights_out = {}

        # indica se o output já foi calculado
        self._output_ready = False
        # indica se o erro já foi calculado
        self._error_ready = False

    # valores da saida atual da camada        
    @property
    def outputs(self):
        
        if not self._output_ready:
            # forward propagation 
            input_sum = 0
            
            # calculo do somatorio de entradas na camada
            for layer in self.weights_in :
                input_sum += self.weights_in[layer].matrix.transpose() * layer.outputs
            batch_size = input_sum.shape[1]
            input_sum += matlib.repmat(self.bias, 1, batch_size) 
            # calculo do output ( y = f(z) )  
            if(self.activation_function == ActivationFuntions.SIGMOID):
                y = sigmoid(input_sum, self.p)
                
            elif(self.activation_function == ActivationFuntions.SOFTMAX):
                aux = np.exp(input_sum)
                y = aux / np.sum(aux, 0)
                
            elif(self.activation_function == ActivationFuntions.LINEAR):
                y = input_sum
            self.outputs = y
            
        return self._outputs
    
    @outputs.setter
    def outputs(self, value):
        self._outputs = value
        self._output_ready = True
        
    # derivada do erro em relação ao input
    @property
    def errors(self):
        if not self._error_ready:
            
            # back propagation
            error_sum = 0
            for layer in self.weights_out :
                error_sum += self.weights_out[layer].matrix * layer.errors
                
            if(self.activation_function == ActivationFuntions.SIGMOID or self.activation_function == ActivationFuntions.SOFTMAX):
                e = np.multiply(sigmoid_derivative(self.outputs, self.p), error_sum)
            elif(self.activation_function == ActivationFuntions.LINEAR):
                e = error_sum
            self.errors = e
            
        return self._errors
   
    @errors.setter
    def errors(self, value):
        self._errors = value
        self._error_ready = True
    
    # atualiza os pesos utilizando o gradiente do erro
    def update_weights(self):
        
        self.bias = self.bias - self.learning_rate * (self.errors.sum(axis=1))
        
        for layer in self.weights_in :
            gradient = layer.outputs * (self.errors.transpose())
            self.weights_in[layer].matrix -= self.learning_rate * gradient
            
