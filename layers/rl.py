import tensorflow as tf
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.resolve().parent))
from layers import Activation


class NoisyDense(tf.keras.layers.Dense):
    def __init__(self, units, 
                 activation=None, 
                 use_bias=True, 
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros', 
                 kernel_regularizer=None, 
                 bias_regularizer=None,
                 activity_regularizer=None, 
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super().__init__(units, 
                         activation=None, 
                         use_bias=True, 
                         kernel_initializer='glorot_uniform',
                         bias_initializer='zeros', 
                         kernel_regularizer=None, 
                         bias_regularizer=None,
                         activity_regularizer=None, 
                         kernel_constraint=None,
                         bias_constraint=None)
    
    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.mu_w = self.add_weight(shape=(input_dim, self.units),
                                    initializer=self.kernel_initializer,
                                    name='mu_weight',
                                    trainable=True,
                                    regularizer=self.kernel_regularizer,
                                    constraint=self.kernel_constraint)
        self.sigma_w = self.add_weight(shape=(input_dim, self.units),
                                       initializer=self.kernel_initializer,
                                       name='sigma_weight',
                                       trainable=True,
                                       regularizer=self.kernel_regularizer,
                                       constraint=self.kernel_constraint)
        self.epsilon_w = self.add_weight(shape=(input_dim, self.units),
                                         initializer='zeros',
                                         name='epsilon_weight')
        
        if self.use_bias:
            self.mu_b = self.add_weight(shape=(self.units, ),
                                        initializer=self.bias_initializer,
                                        name='mu_bias',
                                        trainable=True,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
            self.sigma_b = self.add_weight(shape=(self.units, ),
                                           initializer=self.bias_initializer,
                                           name='sigma_bias',
                                           trainable=True,
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint)
            self.epsilon_b = self.add_weight(shape=(self.units, ),
                                             initializer='zeros',
                                             name='epsilon_bias')
        self.reset_noise()

    def call(self, inputs, 
             training=None,
             mask=None):
        x = inputs
        w = self.mu_w + self.sigma_w * self.epsilon_w
        b = self.mu_b + self.sigma_b * self.epsilon_b if self.use_bias else tf.constant(0., dtype=tf.float32)
        y = tf.matmul(x, w) + b
        return Activation(self.activation)(y)

    def reset_noise(self):
        self.epsilon_w.assign(tf.random.normal(shape=self.epsilon_w.shape))
        if self.use_bias:
            self.epsilon_b.assign(tf.random.normal(shape=self.epsilon_b.shape))
