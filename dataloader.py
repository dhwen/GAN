import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class DataLoader:

    def __init__(self):
        pass

    def generate_true_samples(self, count, dims, type):

        if type == "numerical":
            true_samples = np.zeros((count,dims[0]))
            for i in range(count):
                true_samples[i] = np.random.randint(100,105,dims[0]) #True samples are randomly generated values from 100 to 105
            return true_samples
        elif type =="mnist":
            data = input_data.read_data_sets("MNIST_data/", one_hot=True)
            true_samples = np.asarray(data.train.next_batch(count)[0])
            np.reshape(true_samples,(count,28,28))
            return true_samples
        else:
            print("Unsupported type")

    def generate_noise_samples(self, count, dims, mu=0, sigma=1):
        noise_samples = np.zeros((count,dims[0]))
        for i in range(count):
            noise_samples[i]= np.random.normal(mu,sigma,dims[0]) #Seed noise for the generative network
        return noise_samples