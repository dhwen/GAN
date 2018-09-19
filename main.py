import tensorflow as tf
import numpy as np
import os
from model import ModelGAN
from dataloader import DataLoader

ckpt_path = 'model_ckpt/'
if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)

loader = DataLoader()
true_samples = loader.generate_true_samples(type="numerical", sample_count=200)
noise_samples = loader.generate_noise_samples(sample_count=100)

net = ModelGAN()

num_train_epochs_outter = np.power(10, 2)
num_train_epochs_gen = 5*np.power(10, 3)
num_train_epochs_disc = np.power(10, 2)

with tf.Session(graph=net.graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

#    if os.path.isfile(ckpt_path + 'GAN.ckpt.meta'):
#        saver.restore(sess, tf.train.latest_checkpoint(ckpt_path))
#        print("Restored ckpt")

    true_labels = np.zeros((len(true_samples), 2))
    true_labels[:, 0] = 1
    false_labels = np.zeros((len(noise_samples), 2))
    false_labels[:, 1] = 1

    noise_labels = np.zeros((len(noise_samples), 2))
    noise_labels[:, 0] = 1

    data_labels = np.zeros((len(noise_samples)+len(true_samples), 2))

    for i in range(num_train_epochs_outter):

        for j in range(num_train_epochs_gen):
            [opt, output, loss] = sess.run([net.opt_gen, net.output_disc, net.loss], feed_dict={net.input_gen: noise_samples, net.label: noise_labels})
            print('Epoch %d %d, training loss for the generative network is %g' % (i, j, loss))

            false_samples = sess.run(net.input_disc, feed_dict={net.input_gen: noise_samples, net.label: noise_labels})

        data = np.concatenate((true_samples, false_samples), axis=0)
        data_labels = np.concatenate((true_labels, false_labels), axis=0)

        idcs_new = np.random.permutation(len(true_samples)+len(noise_samples))

        data = data[idcs_new - 1]
        data_labels = data_labels[idcs_new - 1]

        for j in range(num_train_epochs_disc):
            [opt, output, loss] = sess.run([net.opt_disc, net.output_disc, net.loss], feed_dict={net.input_disc: data, net.label: data_labels})
            print('Epoch %d %d, training loss for the discriminative network is %g' % (i, j, loss))

        if i % 9 == 0:
            saver.save(sess, ckpt_path + "GAN.ckpt")