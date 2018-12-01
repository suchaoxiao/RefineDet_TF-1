import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim


def safe_divide(numerator, denominator, name):
    """Divides two values, returning 0 if the denominator is <= 0.
    Args:
      numerator: A real `Tensor`.
      denominator: A real `Tensor`, with dtype matching `numerator`.
      name: Name for the returned op.
    Returns:
      0 if `denominator` <= 0, else `numerator` / `denominator`
    """
    return tf.where(
        math_ops.greater(denominator, 0),
        math_ops.divide(numerator, denominator),
        tf.zeros_like(numerator),
        name=name)

def cummax(x, reverse=False, name=None):
    """Compute the cumulative maximum of the tensor `x` along `axis`. This
    operation is similar to the more classic `cumsum`. Only support 1D Tensor
    for now.

    Args:
    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
       `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
       `complex128`, `qint8`, `quint8`, `qint32`, `half`.
       axis: A `Tensor` of type `int32` (default: 0).
       reverse: A `bool` (default: False).
       name: A name for the operation (optional).
    Returns:
    A `Tensor`. Has the same type as `x`.
    """
    with ops.name_scope(name, "Cummax", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        # Not very optimal: should directly integrate reverse into tf.scan.
        if reverse:
            x = tf.reverse(x, axis=[0])
        # 'Accumlating' maximum: ensure it is always increasing.
        cmax = tf.scan(lambda a, y: tf.maximum(a, y), x,
                       initializer=None, parallel_iterations=1,
                       back_prop=False, swap_memory=False)
        if reverse:
            cmax = tf.reverse(cmax, axis=[0])
        return cmax

def conv2d(input, output_chn, kernel_size, stride=1, use_bias=True, name='conv'):
    return tf.layers.conv2d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            padding="same", data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(
                                0.0, 0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)

def conv3d(input, output_chn, kernel_size, stride=1, use_bias=True, name='conv'):
    return tf.layers.conv3d(inputs=input, filters=output_chn, kernel_size=kernel_size, strides=stride,
                            padding="same", data_format='channels_last',
                            kernel_initializer=tf.truncated_normal_initializer(
                                0.0, 0.01),
                            kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)

def deconv2d(input, output_chn, use_bias=True, kernel_size=2, name='deconv'):
    conv = tf.layers.conv2d_transpose(inputs=input, filters=output_chn, kernel_size=kernel_size,
                                      strides=[2, 2],
                                      padding="same", data_format='channels_last',
                                      kernel_initializer=tf.truncated_normal_initializer(
                                          0.0, 0.01),
                                      kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)
    return conv

def deconv3d(input, output_chn, use_bias=True, kernel_size=2, name='deconv'):
    conv = tf.layers.conv3d_transpose(inputs=input, filters=output_chn, kernel_size=kernel_size,
                                      strides=[2, 2, 2],
                                      padding="same", data_format='channels_last',
                                      kernel_initializer=tf.truncated_normal_initializer(
                                          0.0, 0.01),
                                      kernel_regularizer=slim.l2_regularizer(0.0005), use_bias=use_bias, name=name)
    return conv

def conv_unit(input, output_chn, kernel_size, stride, bn_epsilon=1e-3, is_training=True, name=None):
    with tf.variable_scope(name):
        conv = conv2d(input, output_chn, kernel_size, stride)
        # with tf.device("/cpu:0"):
        bn = tf.layers.batch_normalization(conv, epsilon=bn_epsilon,
                                           training=is_training, name="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu

def conv3d_unit(input, output_chn, kernel_size, stride, bn_epsilon, is_training, name):
    with tf.variable_scope(name):
        conv = conv3d(input, output_chn, kernel_size, stride)
        # with tf.device("/cpu:0"):
        bn = tf.layers.batch_normalization(conv, epsilon=bn_epsilon,
                                           training=is_training, name="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu

def deconv_unit(input, output_chn, bn_epsilon=1e-3, is_training=True, name=None):
    with tf.variable_scope(name):
        conv = deconv2d(input, output_chn)
        # with tf.device("/cpu:0"):
        bn = tf.layers.batch_normalization(conv, epsilon=bn_epsilon,
                                           training=is_training, name="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu

def deconv3d_unit(input, output_chn, bn_epsilon, is_training, name):
    with tf.variable_scope(name):
        conv = deconv3d(input, output_chn)
        # with tf.device("/cpu:0"):
        bn = tf.layers.batch_normalization(conv, epsilon=bn_epsilon,
                                           training=is_training, name="batch_norm")
        relu = tf.nn.relu(bn, name='relu')
    return relu

def res_unit(inputI, n_in, n_out, bn_epsilon=1e-3, stride=1, is_training=True, name='residule_unit'):
    '''
    2D resnet module for the model
    '''
    with tf.variable_scope(name):
        if stride != 1 or n_out != n_in:
            shortcut1 = conv2d(input=inputI, output_chn=n_out,
                               kernel_size=1, stride=stride, name="conv_shortcut")
            shortcut = tf.layers.batch_normalization(shortcut1, epsilon=bn_epsilon,
                                                     training=is_training, name="bn_shortcut")
        else:
            shortcut = inputI

        residual = shortcut
        out = conv_unit(input=inputI, output_chn=n_out, kernel_size=3,
                           stride=1, is_training=is_training, name="pr_cbr_out")
        out = conv2d(input=out, output_chn=n_out,
                     kernel_size=3, stride=stride, name="pr_conv_out")
        out = tf.layers.batch_normalization(out, epsilon=bn_epsilon,
                                            training=is_training, name="pr_bn_out")
        out += residual
        out = tf.nn.relu(out, name='pres_relu_out')
    return out

def res3d_unit(inputI, n_in, n_out, bn_epsilon=1e-3, stride=1, is_training=True, name='residule_unit'):
    '''
    3D resnet module for the model
    '''
    with tf.variable_scope(name):
        if stride != 1 or n_out != n_in:
            shortcut1 = conv3d(input=inputI, output_chn=n_out,
                               kernel_size=1, stride=stride, name="conv_shortcut")
            shortcut = tf.layers.batch_normalization(shortcut1, epsilon=bn_epsilon,
                                                     training=is_training, name="bn_shortcut")
        else:
            shortcut = inputI

        residual = shortcut
        out = conv_unit(input=inputI, output_chn=n_out, kernel_size=3,
                           stride=1, is_training=is_training, name="pr_cbr_out")
        out = conv3d(input=out, output_chn=n_out,
                     kernel_size=3, stride=1, name="pr_conv_out")
        out = tf.layers.batch_normalization(out, epsilon=bn_epsilon,
                                            training=is_training, name="pr_bn_out")
        out += residual
        out = tf.nn.relu(out, name='pres_relu_out')
    return out

