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
    def load(self, file):
        for line in file:
            aux = line.split(',')
            self.add_training_case(*[int(c) for c in aux])    
    def split(self, *subsets_sizes):
        subsets = [DataSet() for _ in subsets_sizes]
        aux = 0
        for i in range(len(subsets_sizes)):
            subsets[i]._data = self._data[:, aux:aux + subsets_sizes[i]]
            aux += subsets_sizes[i]
        return tuple(subsets)
    def randomize(self):
        np.random.shuffle(np.transpose(self._data))
    
    def __len__(self):
        return self._data.shape[1]

class ErrorFunctions(Enum):
    SQUARED_ERROR = 0
    CROSS_ENTROPY = 1
    
