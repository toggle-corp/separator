import tensorflow as tf


def weight_variable(shape):
    """Create weight variable with noisy initialization"""
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))


def bias_variable(shape):
    """Create a bias variable"""
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    """2d convolution with stride size 1 and zero padding"""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def deconv2d(x, W, output_shape):
    """Transpose of 2d convolution"""
    return tf.nn.conv2d_transpose(x, W, output_shape,
                                  strides=[1, 1, 1, 1])


def max_pool(x):
    """Max pooling over 2x2 blocks"""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


class Network:
    def __init__(self, input_shape):
        """Create a neural network with given input size"""
        self.x = tf.placeholder(tf.float32, input_shape)
        self.keep_prob = tf.placeholder(tf.float32)
        self.last = self.x

    def add_reshape(self, shape):
        """Add a reshape layer to reshape data"""
        self.last = tf.reshape(self.last, shape)
        return self

    def add_conv2d(self, filter_height, filter_width,
                   in_channels, out_channels):
        """Add a convolutional neural network layer"""

        W = weight_variable([filter_height, filter_width,
                             in_channels, out_channels])
        b = bias_variable([out_channels])

        self.last = tf.nn.relu(conv2d(self.last, W) + b)
        return self

    def add_maxpool(self):
        """Add maxpooling layer"""
        self.last = max_pool(self.last)
        return self

    def add_dense(self, in_channels, out_channels):
        """Add a dense neural network layer"""
        W = weight_variable([in_channels, out_channels])
        b = bias_variable([out_channels])
        self.last = tf.nn.relu(tf.matmul(self.last, W) + b)
        return self

    def add_dropout(self):
        """Add dropout layer, useful for reducing overfitting
        in dense layer
        """
        self.last = tf.nn.dropout(self.last, self.keep_prob)
        return self

    def add_inverse_conv2d(self, filter_height, filter_width,
                           out_channels, in_channels, out_shape):
        """Add transpose of convolutional layer"""

        W = weight_variable([filter_height, filter_width,
                             out_channels, in_channels])
        b = bias_variable([out_channels])

        self.last = tf.nn.relu(deconv2d(self.last, W, out_shape) + b)
        return self

    def add_resize(self, shape):
        """Add resizing layer"""
        self.last = tf.image.resize_images(self.last, shape)
        return self

    def add_output(self, shape):
        """Add layer of given size"""
        self.y = self.last
        self.y_ = tf.placeholder(tf.float32, shape)

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=self.y, labels=self.y_))
        self.train_step = tf.train.AdamOptimizer(1e-4)\
            .minimize(cross_entropy)

        correct_prediction = tf.equal(
            tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        return self

    def train(self, xs, ys):
        """Train the neural network with given inputs, outputs"""
        self.sess.run(self.train_step, feed_dict={
            self.x: xs, self.y_: ys, self.keep_prob: 0.5
        })

    def evaluate(self, xs, ys):
        """Evaluate the accuracy of neural network
        with given inputs, outputs
        """
        return self.sess.run(self.accuracy, feed_dict={
            self.x: xs, self.y_: ys, self.keep_prob: 1.0
        })

# EOF
