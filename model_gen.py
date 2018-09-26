import tensorflow as tf
import numpy as np
from layers import *

class ModelGen:
    def __init__(self, input_dims, output_dims, type):
        if type == "CNN":
            self.build_model_CNN(input_dims, output_dims)
        elif type == "DNN":
            self.build_model_DNN(input_dims, output_dims)
        else:
            print("Unsupported NN Architecture!")

    def build_model_CNN(self, input_dims, output_dims):
        pass
        if (len(input_dims) != 1 or len(output_dims) != 3):
            print("Invalid number of dimensions for CNN on input/output (Need 1/3)")
            return

        self.input = tf.placeholder(dtype=tf.float32, shape=(None, input_dims[0]), name="Input")
        fc1 = DenseStack(self.input, 12, id=1)
        fc2 = DenseStack(fc1, 16, id=2)
        fc3 = DenseStack(fc2, 16, id=3)
        fc4 = DenseStack(fc3, 24, id=4)
        fc5 = DenseStack(fc4, 24, id=5)
        fc6 = DenseStack(fc5, output_dims[0]*output_dims[1]*output_dims[2], id=6)
        self.output = tf.reshape(fc6, [-1, *output_dims])

    def build_model_DNN(self, input_dims, output_dims):
        if (len(input_dims) != 1 or len(output_dims) != 1):
            print("Invalid number of dimensions for DNN on input/output (Need 1/1)")
            return

        self.input = tf.placeholder(dtype=tf.float32, shape=(None, input_dims[0]), name="Input")
        fc1 = DenseStack(self.input, 10, id=1)
        fc2 = DenseStack(fc1, 11, id=2)
        fc3 = DenseStack(fc2, 11, id=3)
        fc4 = DenseStack(fc3, 10, id=4)
        fc5 = DenseStack(fc4, 9, id=5)
        fc6 = DenseStack(fc5, 8, id=6)
        fc7 = DenseStack(fc6, 8, id=7)
        self.output = tf.layers.dense(fc7, output_dims[0], name="Output")