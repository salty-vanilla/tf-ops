import tensorflow as tf
import tensorflow.keras.backend as K


class PixelNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        x = inputs
        return x / tf.sqrt(tf.reduce_mean(tf.square(x),
                                          axis=-1,
                                          keepdims=True))

    def compute_output_shape(self, input_shape):
        return input_shape


class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self, beta_initializer='zeros',
                 gamma_initializer='ones'):
        super().__init__()
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1]),
                                     name='gamma',
                                     initializer=self.gamma_initializer)
        self.beta = self.add_weight(shape=(input_shape[-1]),
                                    name='beta',
                                    initializer=self.beta_initializer)

    def call(self, inputs, **kwargs):
        x = inputs
        mean, var = tf.nn.moments(x, [1, 2], keepdims=True)
        return tf.nn.batch_normalization(x,
                                         mean,
                                         var,
                                         offset=self.beta,
                                         scale=self.gamma,
                                         variance_epsilon=K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape


class LayerNorm(tf.keras.layers.Layer):
    def __init__(self, begin_norm_axis=1,
                 begin_params_axis=-1,
                 beta_initializer='zeros',
                 gamma_initializer='ones'):
        super().__init__()
        self.begin_norm_axis = begin_norm_axis
        self.begin_param_axis = begin_params_axis
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer

    def build(self, input_shape):
        params_shape = input_shape[self.begin_param_axis:]
        self.input_rank = input_shape.ndims
        if self.begin_norm_axis < 0:
            self.begin_norm_axis = self.input_rank + self.begin_norm_axis

        self.gamma = self.add_weight(shape=params_shape,
                                     name='gamma',
                                     initializer=self.gamma_initializer)
        self.beta = self.add_weight(shape=params_shape,
                                    name='beta',
                                    initializer=self.beta_initializer)

    def call(self, inputs, **kwargs):
        x = inputs
        norm_axes = list(range(self.begin_norm_axis, self.input_rank))
        mean, var = tf.nn.moments(inputs, norm_axes, keepdims=True)
        return tf.nn.batch_normalization(x,
                                         mean,
                                         var,
                                         offset=self.beta,
                                         scale=self.gamma,
                                         variance_epsilon=K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape


class SpectralNorm(tf.keras.Model):
    def __init__(self, layer,
                 power_iteration=1,
                 initializer=tf.keras.initializers.TruncatedNormal(stddev=0.2)):
        super().__init__()
        self.layer = layer
        self.power_iteration = power_iteration
        self.initializer = initializer
        self._is_set = False

        if hasattr(self.layer, 'filters'):
            d = self.layer.filters
        elif hasattr(self.layer, 'units'):
            d = self.layer.units
        else:
            raise AttributeError
        self.d = d

        self.u = tf.Variable(self.initializer(shape=(1, d)),
                             trainable=False,
                             name='u')
        self.sigma = tf.Variable(0.,
                                 trainable=False,
                                 name='sigma')

    def call(self, inputs,
             training=None,
             **kwargs):
        with tf.init_scope():
            if not self._is_set:
                self.layer(inputs)
                self._is_set = True
            if training:
                w = self.layer.kernel
                w_shape = w.shape.as_list()
                w = tf.reshape(w, (-1, self.d))

                u_hat = self.u
                v_hat = None

                for _ in range(self.power_iteration):
                    v_ = tf.matmul(u_hat, w, transpose_b=True)
                    v_hat = self.l2_normalize(v_)

                    u_ = tf.matmul(v_hat, w)
                    u_hat = self.l2_normalize(u_)

                # u_hat = tf.stop_gradient(u_hat)
                # v_hat = tf.stop_gradient(v_hat)

                sigma = tf.matmul(tf.matmul(v_hat, w), u_hat, transpose_b=True)
                sigma = tf.reshape(sigma, ())
                self.u.assign(u_hat)
                self.sigma.assign(sigma)

        return self.layer(inputs, **kwargs) / (self.sigma + K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def l2_normalize(x):
        return x / tf.sqrt(tf.math.maximum(tf.reduce_sum(x**2), tf.keras.backend.epsilon()))


class PositionalNormalization(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        x = inputs
        mean, var = tf.nn.moments(x, [3], keep_dims=True)
        std = tf.sqrt(var + K.epsilon())
        output = (x - mean) / std
        return output, mean, std

    def compute_output_shape(self, input_shape):
        return input_shape


class MomentShortcut(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, inputs, **kwargs):
        x, beta, gamma = inputs
        return x*gamma + beta

    def compute_output_shape(self, input_shape):
        return input_shape


class SwitchNorm(tf.keras.layers.Layer):
    def __init__(self, momentum=0.99,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_var_initializer='ones'):
        super().__init__()
        self.momentum = tf.constant(momentum, dtype=tf.float32)
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.moving_mean_initializer = moving_mean_initializer
        self.moving_var_initializer = moving_var_initializer        

    def build(self, input_shape):
        self.gamma = self.add_weight(shape=(input_shape[-1]),
                                     name='gamma',
                                     initializer=self.gamma_initializer)
        self.beta = self.add_weight(shape=(input_shape[-1]),
                                    name='beta',
                                    initializer=self.beta_initializer)

        self.mean_weight = self.add_weight(shape=(3),
                                           name='mean_weight',
                                           initializer='ones')
        self.var_weight = self.add_weight(shape=(3),
                                          name='var_weight',
                                          initializer='ones')

        self.moving_mean =  self.add_weight(shape=(*[1 for _ in range(len(input_shape)-1)], input_shape[-1]),
                                            name='moving_mean',
                                            initializer=self.moving_mean_initializer,
                                            trainable=False)
        self.moving_var =  self.add_weight(shape=(*[1 for _ in range(len(input_shape)-1)], input_shape[-1]),
                                           name='moving_var',
                                           initializer=self.moving_var_initializer,
                                           trainable=False)

    def call(self, inputs, 
             training=None,
             **kwargs):
        x = inputs
        if training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims=True)
            tf.assign(self.moving_mean,
                      self.momentum*self.moving_mean + (1.-self.momentum)*batch_mean)
            tf.assign(self.moving_var,
                      self.momentum*self.moving_var + (1.-self.momentum)*batch_var) 
        else:
            batch_mean = self.moving_mean
            batch_var = self.moving_var
                            
        instance_mean, instance_var = tf.nn.moments(x, [1, 2], keep_dims=True)
        layer_mean, layer_var = tf.nn.moments(x, [1, 2, 3], keep_dims=True)

        mean_weight = tf.nn.softmax(self.mean_weight)
        var_weight = tf.nn.softmax(self.var_weight)

        mean = mean_weight[0]*batch_mean + mean_weight[1]*instance_mean + mean_weight[2]*layer_mean
        var = var_weight[0]*batch_var + var_weight[1]*instance_var + var_weight[2]*layer_var
        return tf.nn.batch_normalization(x,
                                         mean,
                                         var,
                                         offset=self.beta,
                                         scale=self.gamma,
                                         variance_epsilon=K.epsilon())

    def compute_output_shape(self, input_shape):
        return input_shape
