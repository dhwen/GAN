import tensorflow as tf

def DenseStack(inputs, nNodes, id):
    with tf.variable_scope("DenseStack" + str(id)):
        fc = tf.layers.dense(inputs, nNodes, name="FC")
        bn = tf.layers.batch_normalization(fc, name="BN")
        leaky_relu = tf.nn.leaky_relu(fc, alpha=0.01, name='LeakyRelu')
    return leaky_relu


def ConvStack(inputs, nChannels, conv_filter_dim, pool_filter_dim, id):
    with tf.variable_scope("ConvStack" + str(id)):
        conv = tf.layers.conv2d(inputs=inputs, filters=nChannels, kernel_size=conv_filter_dim, name="Conv")
        pooling = tf.layers.max_pooling2d(inputs=conv, pool_size=pool_filter_dim, strides=[1, 1], name="Pooling")
        leaky_relu = tf.nn.leaky_relu(features=pooling, alpha=0.01, name="LeakyRelu")
        output = tf.layers.batch_normalization(inputs=leaky_relu, name="BN")
    return output