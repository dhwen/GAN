import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class DataLoader:

    def __init__(self):
        pass

    def generate_true_samples(self, type, sample_count, dim):

        if type == "numerical":
            true_samples = np.zeros((sample_count,dim))
            for i in range(sample_count):
                true_samples[i] = np.random.randint(100,105,dim) #True samples are randomly generated values from 100 to 105
            return true_samples
        elif type =="mnist":
            data = input_data.read_data_sets("MNIST_data/", one_hot=True)
            true_samples = np.asarray(data.train.next_batch(sample_count)[0])
            np.reshape(true_samples,(sample_count,28,28))
            return true_samples
        else:
            print("Unsupported type")

    def generate_noise_samples(self, sample_count, dim=10, mu=0, sigma=1):
        noise_samples = np.zeros((sample_count,dim))
        for i in range(sample_count):
            noise_samples[i]= np.random.normal(mu,sigma,dim) #Seed noise for the generative network
        return noise_samples