import tensorflow as tf
from model_disc import ModelDisc
from model_gen import ModelGen

class GAN:

    def __init__(self, input_dims, output_dims, type="DNN"):
        self.type = type
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_gen(input_dims, output_dims, type)
            self.build_disc(type)
            self.build_backprop()

    def build_gen(self, input_dim, output_dim, type):
        with tf.variable_scope("Generative"):
            model_generative = ModelGen(input_dim, output_dim, type)
            self.input_gen = model_generative.input
            self.output_gen = model_generative.output

    def build_disc(self, type):
        with tf.variable_scope("Discriminative"):
            model_discrimintaive = ModelDisc(self.output_gen, type)
            self.input_disc = model_discrimintaive.input
            self.output_disc = model_discrimintaive.output

    def build_backprop(self):
        self.label = tf.placeholder(dtype="float32", shape=(None, 2), name="Label")
        self.loss = tf.losses.softmax_cross_entropy(self.label,self.output_disc)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

        tvars = tf.trainable_variables()
        g_vars_gen = [var for var in tvars if "Generative" in var.name]
        g_vars_disc = [var for var in tvars if "Discriminative" in var.name]

        self.opt_gen = self.optimizer.minimize(self.loss, var_list=g_vars_gen)
        self.opt_disc = self.optimizer.minimize(self.loss, var_list=g_vars_disc)