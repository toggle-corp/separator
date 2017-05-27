import tensorflow as tf
from nn.network import Network, weight_variable, bias_variable


class TestNetwork:
    def __init__(self, batches, height, width, channels=1):
        nn = Network([batches, height, width, channels])

        nn.add_conv2d(3, 1, 10)
        nn.add_max_pool()

        nn.add_conv2d(3, 10, 10)
        nn.add_max_pool()

        height = int(height / 4)
        width = int(width / 4)

        nn.add_reshape([batches, height * width * 10])
        nn.add_dense(height * width * 10, 256)

        nn.add_dense(256, height * width * 10)
        nn.add_reshape([batches, height, width, 10])

        height = height * 2
        width = width * 2

        nn.add_resize([height, width])
        nn.add_conv2d_transpose(3, 10, 10, [batches, height, width, 10])

        height = height * 2
        width = width * 2

        nn.add_resize([height, width])
        nn.add_conv2d_transpose(3, 1, 10, [batches, height, width, channels])

        # nn.last = tf.multiply(nn.x, tf.minimum(nn.last, 1))

        nn.add_output([batches, height, width, channels])

        self.nn = nn
        self.nn.initialize_variables()

    def train(self, xs, ys):
        self.nn.train(xs, ys)

    def evaluate(self, xs, ys):
        return self.nn.evaluate(xs, ys)


# EOF
