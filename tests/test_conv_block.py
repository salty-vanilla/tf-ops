import tensorflow as tf
import os
import sys
import pathlib
current_dir = pathlib.Path(__file__).resolve().parent
sys.path.append(os.path.join(current_dir, '../'))
from blocks import ConvBlock


conv = ConvBlock(32,
                 kernel_size=(3, 3),
                 activation_=None,
                 dilation_rate=(1, 1),
                 sampling='same',
                 normalization='batch',
                 spectral_norm=True)
x = tf.random.normal(shape=(10, 32, 32, 3))

y = conv(x, training=True)
print(y)
