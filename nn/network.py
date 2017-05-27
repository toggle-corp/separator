import tensorflow as tf


def weight_variable(shape):
    """Create weight variable with noisy initialization"""
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    """Create a bias variable"""
    return tf.Variable(tf.constant(0.1, shape=shape))


class Network:
    def __init__(self, input_shape):
        """Create a neural network with given input size"""
        self.x = tf.placeholder(tf.float32, input_shape)
        self.last = self.x

    def add_reshape(self, shape):
        self.last = tf.reshape(self.last, shape)
        return self

    def add_conv2d(self, filter_size, in_channels, out_channels,
                   strides=[1, 1, 1, 1], padding='SAME'):
        """Add a convolutional layer"""

        W = weight_variable([filter_size, filter_size,
                             in_channels, out_channels])
        b = bias_variable([out_channels])

        self.last = tf.nn.relu(tf.nn.conv2d(
            self.last, W, strides=strides, padding=padding) + b)
        return self

    def add_conv2d_transpose(self, filter_size, in_channels,
                             out_channels, out_shape,
                             strides=[1, 1, 1, 1], padding='SAME'):
        """Add a transpose of convolutional layer"""

        W = weight_variable([filter_size, filter_size,
                             in_channels, out_channels])
        b = bias_variable([out_channels])

        self.last = tf.nn.relu(tf.nn.conv2d_transpose(
            self.last, W, out_shape,
            strides=strides, padding=padding) + b)

        return self

    def add_max_pool(self, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                     padding='SAME'):
        """Add a max pooling layer"""
        self.last = tf.nn.max_pool(self.last, ksize, strides, padding)
        return self

    def add_resize(self, shape):
        """Add a layer to resize input"""
        self.last = tf.image.resize_images(self.last, shape)
        return self

    def add_dense(self, in_channels, out_channels):
        W = weight_variable([in_channels, out_channels])
        b = bias_variable([out_channels])
        self.last = tf.nn.relu(tf.matmul(self.last, W) + b)

    def add_output(self, shape, learning_rate=0.01):
        """Add output layer and complete network"""
        self.y = self.last
        self.y_ = tf.placeholder(tf.float32, shape)

        self.loss = tf.reduce_sum(tf.square(self.y_ - self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate)\
            .minimize(self.loss)

        self.sess = tf.Session()
        return self

    def initialize_variables(self):
        self.sess.run(tf.global_variables_initializer())
        return self

    def train(self, xs, ys):
        """Train the neural network with given inputs, outputs"""
        self.sess.run(self.optimizer, feed_dict={
            self.x: xs, self.y_: ys,
        })

    def evaluate(self, xs, ys):
        """Evaluate loss of network based on given inputs, outputs"""
        return self.sess.run(self.loss, feed_dict={
            self.x: xs, self.y_: ys,
        })

    def run(self, xs):
        """Run the network for given inputs"""
        return self.sess.run(self.y, feed_dict={
            self.x: xs,
        })


# EOF
