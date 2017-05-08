import os

# import matplotlib.pyplot as plt
import numpy as np

from nn.separator_network import SeparatorNetwork
from sound.train_data import TrainData


# Tensorflow log levels
# 0 : show all
# 1 : hide info messages
# 2 : hide warning and info messages
# 4 : hide errors, warning and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


batches = 100
samples = 100
nn = SeparatorNetwork(batches, samples)
nn.load('test1')
print("Succesfully created a neural network")


# Read in input and output samples
train_data = TrainData(samples)

# train_data.add_test_files('train_files/crosby/in.wav',
#                           'train_files/crosby/out.wav')
# print("Train files loaded: crosby")

train_data.add_test_files('train_files/happy/in.wav',
                          'train_files/happy/out.wav')
print("Train files loaded: happy")

train_data.add_test_files('train_files/roam/in.wav',
                          'train_files/roam/out.wav')
print("Train files loaded: roam")


# Start the training
print('Training')

for i in range(501):
    xs, ys = train_data.get_random(batches)
    nn.train(xs, ys)

    # Every 100th step, print out accuracy of neural network
    # based on the current batch of test data
    if i % 100 == 0:
        print("Step {} : accuracy: {}".format(i, nn.evaluate(xs, ys)))


nn.save('test1')
# plt.plot(input_data, 'b')
# plt.plot(output_data, 'g')
# plt.show()
