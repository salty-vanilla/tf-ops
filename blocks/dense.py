import tensorflow as tf
import os
import sys
import pathlib

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(os.path.join(current_dir, '../'))

from ops.layers.activations import activation
from ops.layers.conv import SubPixelConv2D
from ops.layers.normalizations import *


class DenseBlock(tf.keras.Model):
    def __init__(self, units,
                 activation_=None,
                 normalization=None,
                 spectral_norm=False,
                 **dense_params):
        super().__init__()
        self.units = units
        dense = tf.keras.layers.Dense(units, **dense_params)

        if spectral_norm:
            self.dense = SpectralNorm(self.dense)
        else:
            self.dense = dense

        # Normalization
        if normalization is not None:
            if normalization == 'batch':
                self.norm = tf.keras.layers.BatchNormalization()
            elif normalization == 'layer':
                self.norm = LayerNorm()
            elif normalization == 'instance':
                self.norm = None
            elif normalization == 'pixel':
                self.norm = None
            else:
                raise ValueError
        else:
            self.norm = None

        self.act = activation_

        self.is_feed_training = spectral_norm

    def call(self, inputs,
             training=None,
             mask=None):
        if self.is_feed_training:
            x = self.dense(inputs, training=training)
        else:
            x = self.dense(inputs)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = activation(x, self.act)
        return x