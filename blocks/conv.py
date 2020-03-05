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
        self.sampling = sampling
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
            conv = tf.keras.layers.Conv2D(filters,
                                          kernel_size,
                                          strides=s,
                                          dilation_rate=dilation_rate,
                                          activation=None,
                                          **conv_params)
        elif sampling == 'deconv':
            conv = tf.keras.layers.Conv2DTranspose(filters,
                                                   kernel_size,
                                                   strides=stride,
                                                   dilation_rate=dilation_rate,
                                                   activation=None,
                                                   **conv_params)
        elif sampling == 'subpixel':
            conv = SubPixelConv2D(filters,
                                  rate=2,
                                  kernel_size=kernel_size,
                                  activation=None,
                                  **conv_params)
        else:
            raise ValueError

        if spectral_norm:
            self.conv = SpectralNorm(conv)
        else:
            self.conv = conv

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
        x = inputs
        if self.sampling == 'up':
            x = tf.keras.layers.UpSampling2D(2)(x)

        if self.is_feed_training:
            x = self.conv(x, training=training)
        else:
            x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x, training=training)
        x = activation(x, self.act)

        if self.sampling == 'max_pool':
            x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        elif self.sampling == 'avg_pool':
            x = tf.keras.layers.AveragePooling2D((2, 2))(x)
        return x


class ResidualBlock(tf.keras.Model):
    def __init__(self, filters,
                 kernel_size=(3, 3),
                 activation_=None,
                 dilation_rate=(1, 1),
                 sampling='same',
                 normalization=None,
                 spectral_norm=False,
                 lr_equalization=False,
                 **conv_params):
        super().__init__()
        self.filters = filters
        self.sampling = sampling
        self.conv1 = ConvBlock(filters,
                               kernel_size,
                               activation_,
                               dilation_rate,
                               'same',
                               normalization,
                               spectral_norm,
                               **conv_params)
        self.conv2 = ConvBlock(filters,
                               kernel_size,
                               None,
                               dilation_rate,
                               sampling,
                               normalization,
                               spectral_norm,
                               **conv_params)
        self.act = activation_
        self.shortcut_conv = ConvBlock(filters,
                                       (1, 1),
                                       spectral_norm=spectral_norm,
                                       **conv_params)

    def call(self, inputs,
             training=None,
             mask=None):
        x = self.conv1(inputs, training=training)
        x = self.conv2(x, training=training)

        if inputs.get_shape().as_list()[-1] != self.filters:
            x += self.shortcut(self.shortcut_conv(inputs))
        else:
            x += self.shortcut(inputs)
        x = activation(x, self.act)
        return x

    def shortcut(self, x):
        if self.sampling in ['deconv', 'up', 'subpixel']:
            return tf.keras.layers.UpSampling2D()(x)
        elif self.sampling in ['max_pool', 'avg_pool', 'stride']:
            return tf.keras.layers.AveragePooling2D()(x)
        else:
            return x
