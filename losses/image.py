import tensorflow as tf
import numpy as np


def make_gauss_kernel(kernel_size, sigma):
    _y, _x = np.mgrid[-kernel_size[1] // 2 + 1: kernel_size[1] // 2 + 1,
                      -kernel_size[0] // 2 + 1: kernel_size[0] // 2 + 1]

    _x = _x.reshape(list(_x.shape) + [1, 1])
    _y = _y.reshape(list(_y.shape) + [1, 1])
    x = tf.constant(_x, dtype=tf.float32)
    y = tf.constant(_y, dtype=tf.float32)

    g = tf.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g /= tf.reduce_sum(g)
    return g


def calc_ssim(y_true,
              y_pred,
              L=1.0,
              K1=0.01,
              K2=0.03,
              kernel_size=(3, 3),
              sigma=1.0,
              return_lcs=False):
    """SSIM
    paper: https://ece.uwaterloo.ca/~z70wang/research/ssim/

    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        L: float hyper parameter of SSIM.
        K1: float hyper parameter of SSIM.
        K2: float hyper parameter of SSIM.
        kernel_size: size of gaussian kernel. (x, y)
        sigma: float parameter of gaussian.
        return_lcs: whether to return (l, c, s)
    # Returns
        return_lcs == False
            4D Tensor (None, h-p, w-p, c).
            Each element of the tensor represents SSIM .

        otherwise
            List of 4D Tensors (None, h-p, w-p, c).
    """

    bs, h, w, c = y_true.get_shape().as_list()

    g_kernel = make_gauss_kernel(kernel_size, sigma)
    g_kernel = tf.tile(g_kernel, (1, 1, c, 1))

    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2
    
    mu_true = tf.nn.depthwise_conv2d(y_true, g_kernel, strides=[1, 1, 1, 1], padding='VALID')
    mu_pred = tf.nn.depthwise_conv2d(y_pred, g_kernel, strides=[1, 1, 1, 1], padding='VALID')
    
    sigma_true = tf.nn.depthwise_conv2d(y_true, g_kernel,
                                        strides=[1, 1, 1, 1], padding='VALID') - mu_true
    sigma_pred = tf.nn.depthwise_conv2d(y_pred, g_kernel,
                                        strides=[1, 1, 1, 1], padding='VALID') - mu_pred    
    mu_true_true = mu_true * mu_true
    mu_pred_pred = mu_pred * mu_pred
    mu_true_pred = mu_true * mu_pred

    sigma_true_true = tf.nn.depthwise_conv2d(y_true * y_true, g_kernel,
                                             strides=[1, 1, 1, 1], padding='VALID') - mu_true_true
    sigma_pred_pred = tf.nn.depthwise_conv2d(y_pred * y_pred, g_kernel,
                                             strides=[1, 1, 1, 1], padding='VALID') - mu_pred_pred
    sigma_true_pred = tf.nn.depthwise_conv2d(y_true * y_pred, g_kernel,
                                             strides=[1, 1, 1, 1], padding='VALID') - mu_true_pred

    if not return_lcs:
        ssim = (2 * mu_true_pred + C1) * (2 * sigma_true_pred + C2)
        ssim /= (mu_true_true + mu_pred_pred + C1) * (sigma_true_true + sigma_pred_pred + C2)
        return ssim
    else:
        l = (2*mu_true*mu_pred + C1) / (mu_true_true + mu_pred_pred + C1)
        c = (2*sigma_true*sigma_pred + C2) / (sigma_true_true + sigma_pred_pred + C2)
        s = (sigma_true_pred + C3) / (sigma_true*sigma_pred + C3)
        return l, c, s


def ssim_loss(y_true,
              y_pred,
              L=1.0,
              K1=0.01,
              K2=0.03,
              kernel_size=(3, 3),
              sigma=1.0):
    """SSIM loss function
    paper: https://ece.uwaterloo.ca/~z70wang/research/ssim/

    # Arguments
        y_true: tensor of true targets.
        y_pred: tensor of predicted targets.
        L: float hyper parameter of SSIM.
        K1: float hyper parameter of SSIM.
        K2: float hyper parameter of SSIM.
        kernel_size: size of gaussian kernel. (x, y)
        sigma: float parameter of gaussian.
    # Returns
        Tensor with one scalar loss entry per sample.
        Each scalar represents "1 - ssim" .
    """

    ssim = calc_ssim(y_true, y_pred,
                     L, K1, K2,
                     kernel_size, sigma)
    return 1. - tf.reduce_mean(ssim, axis=[1, 2, 3])


def make_gaussian_pyramid(x,
                          max_level=5,
                          kernel_size=(3, 3),
                          sigma=1.0,
                          gaussian_iteration=1):
    bs, h, w, c = x.get_shape().as_list()
    pyramid = [x]
    g_kernel = make_gauss_kernel(kernel_size, sigma)
    g_kernel = tf.tile(g_kernel, (1, 1, c, 1))

    for level in range(max_level):
        current = pyramid[-1]
        downsampled = tf.nn.avg_pool(current, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        filtered = downsampled
        for _ in range(gaussian_iteration):
            filtered = tf.nn.depthwise_conv2d(filtered, g_kernel,
                                              strides=[1, 1, 1, 1], padding='SAME')
        pyramid.append(filtered)
    return pyramid


def make_laplacian_pyramid(x,
                           max_level=5,
                           kernel_size=(3, 3),
                           sigma=1.0,
                           gaussian_iteration=1):
    g_pyr = make_gaussian_pyramid(x, max_level, kernel_size, sigma, gaussian_iteration)
    l_pyr = []
    for level in range(max_level):
        high_reso = g_pyr[level]
        low_reso = g_pyr[level + 1]

        bs, h, w, c = high_reso.get_shape().as_list()
        up_low_reso = tf.image.resize_bilinear(low_reso, size=(w, h))

        diff = high_reso - up_low_reso
        l_pyr.append(diff)
    return l_pyr


def lap1_loss(y_true,
              y_pred,
              max_level=5,
              kernel_size=(3, 3),
              sigma=1.0,
              gaussian_iteration=1):
    true_pyr = make_laplacian_pyramid(y_true,
                                      max_level=max_level,
                                      kernel_size=kernel_size,
                                      sigma=sigma,
                                      gaussian_iteration=gaussian_iteration)
    pred_pyr = make_laplacian_pyramid(y_pred,
                                      max_level=max_level,
                                      kernel_size=kernel_size,
                                      sigma=sigma,
                                      gaussian_iteration=gaussian_iteration)

    diffs = []
    for t, p in zip(true_pyr, pred_pyr):
        d = tf.reduce_mean(tf.abs(t - p), axis=[1, 2, 3])
        diffs.append(tf.expand_dims(d, axis=-1))
    diffs = tf.concat(diffs, axis=-1)
    diffs = tf.reduce_mean(diffs, axis=-1)
    return diffs