'''
Created on 19 de jan de 2017

@author: aldo
'''

from matplotlib import pyplot as plt

from annpy.training import DataSet
from annpy.neural_nets import EchoStateNetwork

input_string = input('digite uma sequencia : ')
sequence = [int(i) for i in input_string.split(' ')]

print(*sequence[:-1], *sequence[1:])

esn = EchoStateNetwork(1, 10, 1)
esn.learning_rate = 0.05
esn.set_input_hidden_weights(0.04)
esn.set_hidden_hidden_weights(0.09, 0.7)
esn.set_hidden_bias(1)
esn.set_hidden_output_weights(0.01)

training_set = DataSet()
training_set.add_training_case(*sequence[:-1], *sequence[1:])

errors = []

esn.train(training_set, 5000, error_list=errors)

plt.plot(errors)
plt.show()

for i in sequence:
    print(esn.output(i))
print('---------------')
esn.reset()
n = sequence[0]
for _ in range(20):
    print(n)
    n = round(float(esn.output(n)[0]))
