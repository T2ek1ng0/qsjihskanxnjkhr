import numpy as np

class Gaussian_noise:
    def __init__(self, begin_std, end_std, mean=0):
        self.begin_std = begin_std
        self.end_std = end_std
        self.mean = mean

    def make_noise(self, curr_fes: int, max_fes, ps):  # fes: (ps,1), 返回 (ps,1) 的 np.ndarray
        fes = np.arange(curr_fes, curr_fes + ps)
        stds = self.begin_std + (self.end_std - self.begin_std) * fes / max_fes
        noise = np.random.normal(self.mean, stds)
        return noise.reshape(-1, 1)

    def make_batch_noise(self, curr_fes: int, max_fes, ps, cost_fes=0):  # noise_matrix: (batch_size,ps)
        batch_size = cost_fes // ps
        fes = np.arange(curr_fes, curr_fes + ps * batch_size).reshape(batch_size, ps)  # (batch_size,ps)
        total_std = self.begin_std + (self.end_std - self.begin_std) * fes / max_fes
        return np.random.normal(self.mean, total_std)




