import tensorflow as tf
import numpy as np
from layers import *

class ModelDisc:
    def __init__(self, input, type):
        if type == "CNN":
            self.build_model_CNN(input)
        elif type == "DNN":
            self.build_model_DNN(input)
        else:
            print("Unsupported NN Architecture!")

    def build_model_DNN(self, input):
        self.input = tf.identity(input, name="Input")
        fc1 = DenseStack(self.input, 9, 1)
        fc2 = DenseStack(fc1, 8, 2)
        fc3 = DenseStack(fc2, 7, 3)
        fc4 = DenseStack(fc3, 6, 4)
        fc5 = DenseStack(fc4, 6, 5)
        fc6 = tf.layers.dense(fc5, 2, name="FCout")
        self.output = tf.nn.leaky_relu(fc6, alpha=0.01, name="Output")

    def build_model_CNN(self, input):
        pass
        self.input = tf.identity(input, name="Input")
        conv1 = ConvStack(self.input, nChannels=16, conv_filter_dim=[3,3], pool_filter_dim=[5,5], id=1)
        conv2 = ConvStack(conv1, nChannels=16, conv_filter_dim=[3,3], pool_filter_dim=[5,5], id=2)
        conv3 = ConvStack(conv2, nChannels=16, conv_filter_dim=[3,3], pool_filter_dim=[5,5], id=3)
        conv4 = ConvStack(conv3, nChannels=16, conv_filter_dim=[3, 3], pool_filter_dim=[5,5], id=4)
        flatten = tf.reshape(conv4, [-1, np.prod(conv4.get_shape().as_list()[1:])])
        fc1 = tf.layers.dense(inputs=flatten, units=2, name="FCout")
        self.output = tf.nn.leaky_relu(fc1, alpha=0.01, name="Output")