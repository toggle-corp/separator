from nn.network import Network
import tensorflow as tf


class TestNetwork:
    def __init__(self, batches, height, width, num_outputs=2):
        nn = Network([batches, height, width, 1])

        with tf.name_scope('convolutional_1'):
            nn.add_conv2d(3, 1, 10)
            nn.add_max_pool()

        with tf.name_scope('convolutional_2'):
            nn.add_conv2d(3, 10, 10)
            nn.add_max_pool()

        height = int(height / 4)
        width = int(width / 4)

        with tf.name_scope('dense_layer_1'):
            nn.add_reshape([batches, height * width * 10])
            nn.add_dense(height * width * 10, 256)

        # From the second dense layer we are going to have
        # multiple output channels

        with tf.name_scope('dense_layer_2'):
            nn.add_dense(256, height * width * 10*num_outputs)
            nn.add_reshape([batches, height, width, 10*num_outputs])

        height = height * 2
        width = width * 2

        with tf.name_scope('deconvolutional_2'):
            nn.add_resize([height, width])
            nn.add_conv2d_transpose(
                3, 10*num_outputs, 10*num_outputs,
                [batches, height, width, 10*num_outputs])

        height = height * 2
        width = width * 2

        with tf.name_scope('deconvolutional_1'):
            nn.add_resize([height, width])
            nn.add_conv2d_transpose(
                3, 1*num_outputs, 10*num_outputs,
                [batches, height, width, num_outputs])

        if num_outputs > 1:
            with tf.name_scope('soft_mask'):
                nn.add_soft_mask(num_outputs)

        with tf.name_scope('output'):
            nn.add_output([batches, height, width, num_outputs])

        self.nn = nn
        self.nn.initialize_variables()

    def train(self, xs, ys):
        self.nn.train(xs, ys)

    def evaluate(self, xs, ys):
        return self.nn.evaluate(xs, ys)

    def log(self, path):
        self.nn.log(path)


# EOF
