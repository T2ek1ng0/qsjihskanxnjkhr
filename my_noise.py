import numpy as np

class Gaussian_noise:
    def __init__(self, begin_std, end_std, mean=0):
        self.begin_std = begin_std
        self.end_std = end_std
        self.mean = mean

    def make_noise(self, curr_fes: list, max_fes):  # curr_fes: ps,返回长度为 ps 的 list
        total_noise = []
        for i in range(len(curr_fes)):
            std = self.begin_std + (self.end_std - self.begin_std) * curr_fes[i] / max_fes
            total_noise.append(np.random.normal(self.mean, std, size=1).item())
        return total_noise
