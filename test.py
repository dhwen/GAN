import tensorflow as tf
import numpy as np
import os
from PIL import Image
from data_loader import DataLoader
from config_utils import ConfigLoader

#file_json = "config_numerical.json"
file_json = "config_mnist.json"
config = ConfigLoader(file_json)

if not os.path.isdir(config.output_folder):
    os.makedirs(config.output_folder)

loader = DataLoader()
noise_samples = loader.generate_noise_samples(count=config.num_test_samples, dims=config.noise_dims)

with tf.Session(graph=tf.Graph()) as sess:
    saver = tf.train.import_meta_graph(config.ckpt_save_path+config.ckpt_file_name+".meta")
    saver.restore(sess, tf.train.latest_checkpoint(config.ckpt_save_path))

    input = sess.graph.get_tensor_by_name("Generative/Input:0")
    output = sess.graph.get_tensor_by_name(config.output_tensor_name)

    false_samples = sess.run(output, feed_dict={input: noise_samples})

    if config.samples_type == "numerical":
        print(false_samples)
    elif config.samples_type == "mnist":
        img_num = 1
        for false_sample in false_samples:
            img = Image.fromarray(np.squeeze((false_sample*255).astype(np.uint8)))
            img.save(config.output_folder + 'im_' + str(img_num)+'.png')
            img_num += 1

