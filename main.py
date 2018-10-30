import tensorflow as tf
import numpy as np
import os
from GAN import GAN
from data_loader import DataLoader
from config_utils import ConfigLoader

file_json = "config_numerical.json"
#file_json = "config_mnist.json"
config = ConfigLoader(file_json)

if not os.path.isdir(config.ckpt_save_path):
    os.makedirs(config.ckpt_save_path)

loader = DataLoader()
noise_samples = loader.generate_noise_samples(count=config.noise_count, dims=config.noise_dims)
true_samples = loader.generate_true_samples(count=config.samples_count, dims=config.samples_dims, type=config.samples_type)

net = GAN(input_dims=config.noise_dims, output_dims=config.samples_dims, type=config.model_type)

with tf.Session(graph=net.graph) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    #if os.path.isfile(config.ckpt_save_path+config.ckpt_file_name+".meta"):
    #    saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_save_path))
    #    print("Restored ckpt file.")

    true_labels = np.zeros((len(true_samples), 2))
    true_labels[:, 0] = 1
    false_labels = np.zeros((len(noise_samples), 2))
    false_labels[:, 1] = 1
    noise_labels = np.zeros((len(noise_samples), 2))
    noise_labels[:, 0] = 1

    data_labels = np.zeros((len(noise_samples)+len(true_samples), 2))

    for i in range(config.num_iter_main):

        false_samples = sess.run(net.output_gen, feed_dict={net.input_gen: noise_samples})

        data = np.concatenate((true_samples, false_samples), axis=0)
        data_labels = np.concatenate((true_labels, false_labels), axis=0)

        idcs_new = np.random.permutation(len(true_samples)+len(noise_samples)) - 1

        data = data[idcs_new]
        data_labels = data_labels[idcs_new]

        for j in range(config.num_iter_disc):
            idcs_batch_disc = (np.random.permutation(len(true_samples) + len(noise_samples)) - 1)[1:config.batch_size_disc]

            [opt, output, loss] = sess.run([net.opt_disc, net.output_disc, net.loss], feed_dict={net.input_disc: data[idcs_batch_disc], net.label: data_labels[idcs_batch_disc]})
            if (j+1) % config.print_interval_loss_disc == 0 :
                print('Iteration %d-%d, training loss for the discriminative network is %g' % (i+1, j+1, loss))

        for j in range(config.num_iter_gen):
            idcs_batch_gen = (np.random.permutation(len(noise_samples)) - 1)[1:config.batch_size_gen]

            [opt, output, loss] = sess.run([net.opt_gen, net.output_disc, net.loss], feed_dict={net.input_gen: noise_samples[idcs_batch_gen], net.label: noise_labels[idcs_batch_gen]})
            if (j+1) % config.print_interval_loss_gen == 0 :
                print('Iteration %d-%d, training loss for the generative network is %g' % (i+1, j+1, loss))

        if (i+1) % config.save_interval_ckpt == 0:
            saver.save(sess, config.ckpt_save_path + config.ckpt_file_name)