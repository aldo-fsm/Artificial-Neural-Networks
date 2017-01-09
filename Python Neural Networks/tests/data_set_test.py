'''
Created on 8 de jan de 2017

@author: aldo
'''
from annpy.training import DataSet

data = DataSet()

data.add_training_case(1, 0, 0, 1)
data.add_training_case(1, 2, 3, 12)
data.add_training_case(1.2, 0.5, 0, 1)
data.add_training_case(0, 0, 1, 1)

m = data.training_matrices(2)
print(m)
print(len(data))
