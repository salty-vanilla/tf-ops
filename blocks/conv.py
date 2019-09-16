import tensorflow as tf
import os
import sys
import pathlib

current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(os.path.join(current_dir, '../'))

from ops.layers.activations import activation
from ops.layers.conv import SubPixelConv2D
from ops.layers.normalizations import *


class ConvBlock(tf.keras.Model):
    def __init__(self, filters,
                 kernel_size=(3, 3),
                 activation_=None,
                 dilation_rate=(1, 1),
                 sampling='same',
                 normalization=None,
                 spectral_norm=False,
                 **conv_params):
        conv_params.setdefault('padding', 'same')
        super().__init__()

        stride = 1 if sampling in ['same',
                                   'subpixel',
                                   'max_pool',
                                   'avg_pool',
                                   'subpixel'] \
            else 2
        if 'stride' in conv_params:
            stride = conv_params['stride']

        # Convolution
        if sampling in ['up', 'max_pool', 'avg_pool', 'same', 'stride']:
            s = stride if sampling == 'stride' else 1
            self.conv = tf.keras.layers.Conv2D(filters,
                                               kernel_size,
                                               strides=s,
                                               dilation_rate=dilation_rate,
                                               activation=None,
                                               **conv_params)
        elif sampling == 'deconv':
            self.conv = tf.keras.layers.Conv2DTranspose(filters,
                                                        kernel_size,
                                                        strides=stride,
                                                        dilation_rate=dilation_rate,
                                                        activation=None,
                                                        **conv_params)
        elif sampling == 'subpixel':
            self.conv = SubPixelConv2D(filters,
                                       rate=2,
                                       kernel_size=kernel_size,
                                       activation=None,
                                       **conv_params)
        else:
            raise ValueError

        if spectral_norm:
            self.conv = SpectralNorm(self.conv)

        # Normalization
        if normalization is not None:
            if normalization == 'batch':
                self.norm = tf.keras.layers.BatchNormalization()
            elif normalization == 'layer':
                self.norm = LayerNorm()
            elif normalization == 'instance':
                self.norm = InstanceNorm()
            elif normalization == 'pixel':
                self.norm = PixelNorm()
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
            x = self.conv(inputs, training=training)
        else:
            x = self.conv(inputs)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = activation(x, self.act)
        return x
