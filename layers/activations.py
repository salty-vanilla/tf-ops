import tensorflow as tf
from tensorflow.python.keras import layers as kl


class Activation(tf.keras.layers.Layer):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def call(self, inputs,
             *args,
             **kwargs):
        x = inputs
        if self.activation == 'lrelu':
            return tf.nn.leaky_relu(x, 0.2)
        elif self.activation == 'swish':
            return x * tf.math.sigmoid(x)
        else:
            return tf.keras.layers.Activation(self.activation)(x)
