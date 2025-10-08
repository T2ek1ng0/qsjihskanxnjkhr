import numpy as np

class Gaussian_noise:
    def __init__(self, begin_std, end_std, mean=0):
        self.begin_std = begin_std
        self.end_std = end_std
        self.mean = mean

    def make_noise(self, curr_fes, max_fes, res):
        std = self.begin_std + (self.end_std - self.begin_std) * curr_fes / max_fes
        noise = np.random.normal(self.mean, std, res.shape)
        return res+noise
