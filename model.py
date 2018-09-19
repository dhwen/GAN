import tensorflow as tf

class ModelGAN:
    def __init__(self, dropout_drop_prob=0):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.build_gen()
            self.build_disc()
            self.build_backprop()

    def build_gen(self):
        with tf.variable_scope("Generative"):
            self.input_gen = tf.placeholder(dtype=tf.float32, shape=(None, 10), name="Input")
            fc1_gen = self.DenseStack(self.input_gen, 10, 1)
            fc2_gen = self.DenseStack(fc1_gen, 11, 2)
            fc3_gen = self.DenseStack(fc2_gen, 11, 3)
            fc4_gen = self.DenseStack(fc3_gen, 10, 4)
            fc5_gen = self.DenseStack(fc4_gen, 9, 5)
            fc6_gen = self.DenseStack(fc5_gen, 8, 6)
            fc7_gen = self.DenseStack(fc6_gen, 8, 7)
            self.output_gen = tf.layers.dense(fc7_gen, 1, name="Output")

    def build_disc(self):
        with tf.variable_scope("Discriminative"):
            self.input_disc = tf.identity(self.output_gen, name="Input")
            fc1_disc = self.DenseStack(self.input_disc, 7, 1)
            fc2_disc = self.DenseStack(fc1_disc, 8, 2)
            fc3_disc = self.DenseStack(fc2_disc, 7, 3)
            fc4_disc = self.DenseStack(fc3_disc, 6, 4)
            fc5_disc = self.DenseStack(fc4_disc, 6, 5)
            fc6_disc = self.DenseStack(fc5_disc, 4, 6)
            fc7_disc = tf.layers.dense(fc6_disc, 2, name="FC")
            self.output_disc = tf.nn.relu(fc7_disc, name="Output")

    def DenseStack(self, inputs, nNodes, id):
        with tf.variable_scope("DenseStack" + str(id)):
            fc = tf.layers.dense(inputs, nNodes, name="FC")
            #bn = tf.layers.batch_normalization(fc, name="BN")
            relu = tf.nn.relu(fc, name='Relu')
        return relu

    def build_backprop(self):
        self.label = tf.placeholder(dtype="float32", shape=(None, 2), name="Label")
        self.loss = tf.losses.softmax_cross_entropy(self.label,self.output_disc)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')

        tvars = tf.trainable_variables()
        g_vars_gen = [var for var in tvars if "Generative" in var.name]
        g_vars_disc = [var for var in tvars if "Discriminative" in var.name]

        self.opt_gen = self.optimizer.minimize(self.loss, var_list=g_vars_gen)
        self.opt_disc = self.optimizer.minimize(self.loss, var_list=g_vars_disc)