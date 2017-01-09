'''
Created on 8 de jan de 2017

@author: aldo
'''

import numpy as np
from enum import Enum

class DataSet:
    
    def add_training_case(self, *values):
        if not hasattr(self, '_data'):
            self._data = np.array([[]])
            self._data.shape = len(values), 0
        self._data = np.c_[self._data, values]
    
    def training_matrices(self, mini_batch_size):
        if len(self) % mini_batch_size == 0:
            matrix = np.matrix(self._data)
            return [matrix[:, i:i + mini_batch_size] for i in range(0, len(self), mini_batch_size)]
        else :
            raise ValueError('Not divisible by {0}'.format(mini_batch_size))
   
    def __str__(self):
        return str(self._data)
    def __len__(self):
        return self._data.shape[1]

class ErrorFunctions(Enum):
    SQUARED_ERROR = 0
    CROSS_ENTROPY = 1
    