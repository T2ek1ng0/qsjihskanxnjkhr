import numpy as np

class Gaussian_noise:
    def __init__(self, begin_std, end_std, mean=0):
        self.begin_std = begin_std
        self.end_std = end_std
        self.mean = mean

    def make_noise(self, curr_fes: list, max_fes, n_problem):  # curr_fes: ps,返回 (ps, n_problem) 的噪声矩阵
        total_noise = []
        for i in range(len(curr_fes)):
            std = self.begin_std + (self.end_std - self.begin_std) * curr_fes[i] / max_fes
            noise = np.random.normal(self.mean, std, n_problem)
            total_noise.append(noise)
        total_noise = np.array(total_noise, dtype=float)
        return total_noise
