'''
Created on 8 de jan de 2017

@author: aldo
'''

import numpy as np

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
    
# class DataSet:
#     
#     def __init__(self, num_input_layers, num_output_layers):
#         # self._num_input = num_input_layers
#         # self._num_output = num_output_layers
#         self._inputs = np.array([[]])
#         self._outputs = np.array([[]])
#         self._inputs.shape = num_input_layers, 0
#         self._outputs.shape = num_output_layers, 0
#     
#     def add_training_case(self, inputs, desired_outputs):
#         self._inputs = np.c_[self._inputs, inputs]
#         self._outputs = np.c_[self._outputs, desired_outputs]
#     
#     def input_matrices(self, mini_batch_size):
#         if len(self) % mini_batch_size == 0:
#             result = []
#             slices = [self._inputs[:, i:i + mini_batch_size] for i in range(0, len(self), mini_batch_size)]
#             for mini_batch in slices:
#                 l = []
#                 for i in range(self._inputs.shape[0]):
#                     l.append(np.matrix([m for m in mini_batch[i]]).transpose())
#                 result.append(l)
#             return result
#         else :
#             raise ValueError('Not divisible by {0}'.format(mini_batch_size))
#    
#     def output_matrices(self, mini_batch_size):
#         if len(self) % mini_batch_size == 0:
#             result = []
#             slices = [self._outputs[:, i:i + mini_batch_size] for i in range(0, len(self), mini_batch_size)]
#             for mini_batch in slices:
#                 l = []
#                 for i in range(self._outputs.shape[0]):
#                     l.append(np.matrix([m for m in mini_batch[i]]).transpose())
#                 result.append(l)
#             return result
#         else :
#             raise ValueError('Not divisible by {0}'.format(mini_batch_size))
#     
#     def __str__(self):
#         return 'Inputs = {0}\nDesired outputs = {1}'.format(str(self._inputs), str(self._outputs))
#     def __len__(self):
#         return self._inputs.shape[1]
#     def __index__(self, index):
#         return self._data[index]
    
