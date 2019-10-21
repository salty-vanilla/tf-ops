import tensorflow as tf
from ops.layers.normalizations import SpectralNorm


class NonLocal2D(tf.keras.Model):
    def __init__(self, filters,
                 initializer=tf.keras.initializers.Zeros()):
        super().__init__()
        self.f = tf.keras.layers.Conv2D(filters//8, 1)
        self.g = tf.keras.layers.Conv2D(filters//8, 1)
        self.h = tf.keras.layers.Conv2D(filters, 1)
        self.conv = tf.keras.layers.Conv2D(filters, 1)
        self.gamma = tf.Variable(0., 
                                 name='gamma',
                                 trainable=True)

        self.f = SpectralNorm(self.f)
        self.g = SpectralNorm(self.g)
        self.h = SpectralNorm(self.h)
        self.conv = SpectralNorm(self.conv)

    def call(self, inputs,
             training=None):
        x = inputs
        flatten = lambda x: tf.reshape(x, (x.shape[0], -1, x.shape[-1]))

        s = tf.matmul(flatten(self.g(x, training=training)),
                      flatten(self.f(x, training=training)),
                      transpose_b=True)
        beta = tf.nn.softmax(s, axis=-1)

        o = tf.matmul(beta, flatten(self.h(x, training=training)))
        o = tf.reshape(o, x.shape)
        o = self.conv(o, training=training)
        return self.gamma*o + x
