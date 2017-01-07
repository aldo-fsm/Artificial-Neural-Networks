'''
Created on 7 de jan de 2017

@author: aldo
'''

import math

class Layer:
    # pesos das entradas
    weights_in = {}
    # pesos das saidas
    weights_out = {}
    visited = False
    
    def __init__(self, number_units):
        self.size = number_units

    # valores da saida atual da camada        
    @property
    def outputs(self):
        if self.visited :
            return self.outputs 
        else :
            '''
        -----------------------------------------
            calcula o output (falta implementar)
        -----------------------------------------            
            '''
            return None
    @outputs.setter
    def outputs(self, value):
        self.outputs = value
        self.visited = True
        
    # derivada do erro em relação ao input
    @property
    def errors(self):
        if self.visited :
            return self.error
        else :
            '''
        -----------------------------------------
            calcula o erro (falta implementar)
        -----------------------------------------            
            '''
            return None
   
    @errors.setter
    def errors(self, value):
        self.errors = value
        self.visited = True
        
class SynapticWeights:
    def __init__(self, weight_matrix):
        self.matrix = weight_matrix
        
def sigmoid(x, a=1):
    return 1 / (1 + math.exp(-a * x))
        
