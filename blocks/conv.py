import tensorflow as tf
from layers.activations import Activation
from layers.conv import SubPixelConv2D
from layers.normalizations import *


class ConvBlock(tf.keras.Sequential):
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

        # Upsampling
        if sampling == 'up':
            self.add(tf.keras.layers.UpSampling2D(2))
    
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
            conv = SpectralNorm(conv)

        self.add(conv)

         # Normalization
        if normalization is not None:
            if normalization == 'batch':
                norm = tf.keras.layers.BatchNormalization()
            elif normalization == 'layer':
                norm = LayerNorm()
            elif normalization == 'instance':
                norm = InstanceNorm()
            elif normalization == 'pixel':
                norm = PixelNorm()
            else:
                raise ValueError
            self.add(norm)

        self.add(Activation(activation_))

        # Pooling
        if sampling == 'max_pool':
            self.add(tf.keras.layers.MaxPooling2D(2))
        elif sampling == 'avg_pool':
            self.add(tf.keras.layers.AveragePooling2D(2))


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
