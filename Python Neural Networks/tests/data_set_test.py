'''
Created on 8 de jan de 2017

@author: aldo
'''
from annpy.training import DataSet

data = DataSet()

data.add_training_case(1, 0, 0, 1)
data.add_training_case(1, 2, 3, 12)
data.add_training_case(1.2, 0.5, 0, 1)
data.add_training_case(0.2, 23, 1, 1)
data.add_training_case(0, 2, 4, 1)
data.add_training_case(123, 3, 1, 1)

data.randomize()
a, b, c = data.split(1, 2, 3)
print(len(a))
print(len(b))
print(len(c))
m = data.training_matrices(2)
print(m)
print(len(data))
