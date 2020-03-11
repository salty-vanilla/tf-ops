import tensorflow as tf
from layers.activations import Activation
from layers.conv import SubPixelConv2D
from layers.normalizations import *


class DenseBlock(tf.keras.Sequential):
    def __init__(self, units,
                 activation_=None,
                 normalization=None,
                 spectral_norm=False,
                 **dense_params):
        super().__init__()
        self.units = units
        dense = tf.keras.layers.Dense(units, **dense_params)

        if spectral_norm:
            dense = SpectralNorm(dense)
        self.add(dense)

        # Normalization
        if normalization is not None:
            if normalization == 'batch':
                norm = tf.keras.layers.BatchNormalization()
            elif normalization == 'layer':
                norm = LayerNorm()
            elif normalization == 'instance':
                norm = None
            elif normalization == 'pixel':
                norm = None
            else:
                raise ValueError
        if norm:
            self.add(norm)

        self.add(Activation(activation_))
