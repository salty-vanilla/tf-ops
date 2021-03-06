import tensorflow as tf


def generator_loss(d_fake,
                   metrics='JSD'):
    if metrics in ['JSD', 'jsd']:
        return tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake),
                                                    logits=d_fake))
    elif metrics in ['WD', 'wd']:
        return -tf.reduce_mean(d_fake)
    elif metrics in ['HINGE', 'hinge']:
        return -tf.reduce_mean(d_fake)
    elif metrics in ['LS', 'ls', 'PearsonChiSquared', 'PCS', 'pcs']:
        return tf.reduce_mean((d_fake-0)**2)/2
    else:
        raise ValueError


def discriminator_loss(d_real,
                       d_fake,
                       metrics='JSD'):
    if metrics in ['JSD', 'jsd']:
        real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real),
                                                    logits=d_real))
        fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake),
                                                    logits=d_fake))
        return real_loss + fake_loss
    elif metrics in ['WD', 'wd']:
        return -(tf.reduce_mean(d_real) - tf.reduce_mean(d_fake))
    elif metrics in ['HINGE', 'hinge']:
        real_loss = -tf.reduce_mean(tf.minimum(tf.constant(0., dtype=tf.float32), 
                                               tf.constant(-1., dtype=tf.float32) + d_real))
        fake_loss = -tf.reduce_mean(tf.minimum(tf.constant(0., dtype=tf.float32), 
                                               tf.constant(-1., dtype=tf.float32) - d_fake))
        return real_loss + fake_loss
    elif metrics in ['LS', 'ls', 'PearsonChiSquared', 'PCS', 'pcs']:
        return tf.reduce_mean((d_real-1.)**2)/2. + tf.reduce_mean((d_fake+1)**2)/2
    else:
        raise ValueError


def gradient_penalty(discriminator,
                     real,
                     fake):
    bs = real.get_shape().as_list()[0]
    uniform = tf.random.uniform if float(tf.__version__[0]) >= 2.0 else tf.random_uniform

    eps = uniform(shape=(bs, ), 
                  minval=0., maxval=1.)
    for _ in range(len(real.get_shape().as_list()) - 1):
        eps = tf.expand_dims(eps, axis=-1)

    reduction_indices = list(range(1, len(real.get_shape().as_list())))

    differences = fake - real
    interpolates = real + (eps * differences)

    with tf.GradientTape() as g:
        g.watch(interpolates)
        y = discriminator(interpolates,
                          training=True)
    grads = g.gradient(y, interpolates)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grads),
                                   axis=reduction_indices))
    gp = tf.reduce_mean(tf.square(slopes - 1.))
    return gp


def discriminator_norm(d_real):
    with tf.name_scope('DiscriminatorNorm'):
        return tf.nn.l2_loss(d_real)


def pull_away(x, eps=1e-8):
    n, d = x.get_shape().as_list()
    i = tf.eye(n)
    inv_i = tf.cast(tf.cast(i - 1, tf.bool), tf.float32)

    denominator = tf.norm(x, axis=-1)[:, None]
    denominator *= tf.norm(x, axis=-1)[None, :]

    numerator = tf.reduce_sum(x[None, :]*x[:, None], axis=-1) * inv_i

    return tf.reduce_sum((numerator/(denominator+eps))**2) / (n*(n-1))
