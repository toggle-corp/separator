import os
import random

# import matplotlib.pyplot as plt
import numpy as np

from nn.network import Network
from sound.sound import Sound


# Tensorflow log levels
# 0 : show all
# 1 : hide info messages
# 2 : hide warning and info messages
# 4 : hide errors, warning and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


batches = 100
samples = 100

# Create a neural network that takes #batches x #samples of
# input data
nn = Network([batches, samples])

# Reshape the input data to 4D tensor
# [batches, height_of_data, width_of_data, num_channels]
nn.add_reshape([batches, 1, samples, 1])

# Add the first convolutional layer
# (filter_height, filter_width, input_channels, output_channels)
nn.add_conv2d(1, 30, 1, 30)\
    .add_maxpool()

# After pooling once, each feature is half the original size
fsize1 = int(samples/2)

# Add the second convolutional layer
nn.add_conv2d(10, 20, 30, 30)\
    .add_maxpool()

# After pooling two times, each feature is 1/4-th original size
fsize2 = int(samples/4)

# Reshape from 4D tensor to flat data
# to pass to the first dense layer
nn.add_reshape([batches, fsize2 * 30])\
    .add_dense(fsize2 * 30, 256)\
    .add_dropout()

# Second dense layer
nn.add_dense(256, fsize2 * 30)\
    .add_dropout()


# Convert flat data back to 4D tensor and
# add transpose of second convolutional layer
# followed by upscaling
nn.add_reshape([batches, 1, fsize2, 30])\
    .add_inverse_conv2d(10, 20, 30, 30, [batches, 1, fsize2, 30])\
    .add_resize([1, fsize1])

# Add transpose of first convolutional layer
# followed by upscaling
nn.add_inverse_conv2d(1, 30, 1, 30, [batches, 1, fsize1, 1])\
    .add_resize([1, samples])

# Reshape the final 4D tensor back to flat data
# and add output layer
nn.add_reshape([batches, samples])\
    .add_output([batches, samples])

print("Succesfully created a neural network")


# Read in input and output samples

test_input = Sound('test_audio/mix.wav')
# input_data = abs(test_input.fft)
# input_data = input_data / 10**7
input_data = test_input.normalized
print("Input file loaded")

test_output = Sound('test_audio/out_pre_post.wav')
# output_data = abs(test_output.fft)
# output_data = output_data / 10**7
output_data = test_output.normalized
print("Output file loaded")


# Note: this might not be the right way to do it
# but make sure both data are of equal size and is a multiple of #samples
# using padding
length = max(len(input_data), len(output_data))
length = int(np.ceil(length / samples) * samples)
input_data = np.lib.pad(
    input_data, (0, length - len(input_data)), 'constant')
output_data = np.lib.pad(
    output_data, (0, length - len(output_data)), 'constant')


# Start the training
print('Training')

# Idea is to transform the input and output data from
# a list of samples to a matrix where each row
# is #samples size
# That way we can pass number of rows simultaneously
# to be processed by the GPU
input_data = input_data.reshape(-1, samples)
output_data = output_data.reshape(-1, samples)

for i in range(10001):
    # Get a random index and get #batches of xs and ys
    index = random.randint(0, len(input_data) - batches)

    xs = input_data[index: index + batches]
    ys = output_data[index: index + batches]

    # Train using these xs and ys
    nn.train(xs, ys)

    # Every 100th step, print out accuracy of neural network
    # based on the current batch of test data
    if i % 100 == 0:
        print("Step {} : accuracy: {}".format(i, nn.evaluate(xs, ys)))

# plt.plot(input_data, 'b')
# plt.plot(output_data, 'g')
# plt.show()
