import os

from .network import Network


class SeparatorNetwork:
    def __init__(self, num_batches, num_input_samples):

        batches = num_batches
        samples = num_input_samples

        # Create a neural network that takes #batches x #samples of
        # input data
        nn = Network([batches, samples])

        # Reshape the input data to 4D tensor
        # [batches, height_of_data, width_of_data, num_channels]
        nn.add_reshape([batches, 1, samples, 1])

        # Add the first convolutional layer
        # (filter_height, filter_width, in_channels, out_channels)
        nn.add_conv2d(1, 30, 1, 30)\
            .add_maxpool()

        # After pooling once, each feature is half the original size
        fsize1 = int(samples/2)

        # Add the second convolutional layer
        nn.add_conv2d(10, 20, 30, 30)\
            .add_maxpool()

        # After pooling two times, each feature is 1/4th original size
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
            .add_inverse_conv2d(10, 20, 30, 30,
                                [batches, 1, fsize2, 30])\
            .add_resize([1, fsize1])

        # Add transpose of first convolutional layer
        # followed by upscaling
        nn.add_inverse_conv2d(1, 30, 1, 30, [batches, 1, fsize1, 1])\
            .add_resize([1, samples])

        # Reshape the final 4D tensor back to flat data
        # and add output layer
        nn.add_reshape([batches, samples])\
            .add_output([batches, samples])

        self.nn = nn

    def train(self, xs, ys):
        self.nn.train(xs, ys)

    def evaluate(self, xs, ys):
        return self.nn.evaluate(xs, ys)

    def load(self, network_name):
        if os.path.exists('saved_networks/' + network_name):
            self.nn.load_variables(
                'saved_networks/' + network_name + '/network.ckpt')
            print("Loaded from network: {}".format(network_name))
        else:
            self.nn.initialize_variables()

    def save(self, network_name):
        if not os.path.exists('saved_networks/' + network_name):
            try:
                os.makedirs('saved_networks/' + network_name)
            except:
                pass

        self.nn.save_variables(
                'saved_networks/' + network_name + '/network.ckpt')
        print("Saved to network: {}".format(network_name))


# EOF
