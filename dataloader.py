import numpy as np

class DataLoader:

    def __init__(self):
        pass

    def generate_true_samples(self, type, sample_count):

        if type == "numerical":
            true_samples = np.zeros((sample_count,1))
            for i in range(sample_count):
                true_samples[i] = 5*np.random.randint(1,10,1) #True samples are randomly generated multiples of 5 bounded at [5,50]
            return true_samples
        else:
            print("Unsupported type")

    def generate_noise_samples(self, sample_count, mu=0, sigma=1, len=10):
        noise_samples = np.zeros((sample_count,len))
        for i in range(sample_count):
            noise_samples[i]= np.random.normal(mu,sigma,len) #Seed noise for the generative network
        return noise_samples