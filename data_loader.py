import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class DataLoader:

    def __init__(self):
        pass

    def generate_true_samples(self, count, dims, type):

        if type == "numerical":
            return np.random.randint(100,105,(count,*dims)) #True samples are randomly generated values from 100 to 105
        elif type =="mnist":
            data = input_data.read_data_sets("MNIST_data/", one_hot=True)
            true_samples = np.reshape(np.asarray(data.train.next_batch(count)[0]),(count,*dims))
            return true_samples
        else:
            print("Unsupported type")

    def generate_noise_samples(self, count, dims, mu=0, sigma=1):
        return np.random.normal(mu,sigma,(count,*dims)) #Seed noise for the generative network