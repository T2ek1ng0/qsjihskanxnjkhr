import numpy as np
from typing import Union

class Sub_Problem_Weight:
    def __init__(self, n_problem, weights, max_fes, rates: Union[list, np.ndarray, float]):
        if isinstance(rates, list):
            rates = np.array(rates, dtype=float)
        elif isinstance(rates, float):
            if not (0 < rates <= 1.0):
                raise ValueError(f"Invalid rate value {rates}, must be in (0, 1].")
            temp_rates = np.arange(rates, 1.0 + 1e-8, rates)
            rates = np.round(np.clip(temp_rates, None, 1.0), 4)
        elif not isinstance(rates, np.ndarray):
            raise TypeError("rates must be list, np.ndarray, or float")
        if isinstance(rates, np.ndarray):
            if rates[-1] < 1.0 - 1e-8:
                rates = np.append(rates, 1.0)
        self.n_problem = n_problem
        self.rates = rates
        self.weights = np.array(weights, dtype=float)  # (ps,n_problem)
        self.ps = self.weights.shape[0]
        self.maxfes = max_fes
        self.pos = np.argmax(self.weights, axis=1)
        self.fes_thresholds = np.sort(self.rates * self.maxfes)
        self.fes_process = -1

    def get_weight(self):
        return self.weights

    def check(self, curr_fes: int):
        old_process = self.fes_process
        curr_process = np.searchsorted(self.fes_thresholds, curr_fes, side='right') - 1
        self.fes_process = max(self.fes_process, curr_process)
        return curr_process > old_process

    def cal_weight(self, curr_fes: int):  # weights: (ps,n_problem)
        if self.check(curr_fes):
            self.weights = np.zeros_like(self.weights)
            self.pos = (self.pos + 1) % self.n_problem  # (ps,)
            self.weights[np.arange(self.ps), self.pos] = 1
        return self.weights

    def cal_batch_weight(self, curr_fes: int, cost_fes, change_flag=False):  # weight_matrix: (batch_size,ps,n_problem)
        batch_size = cost_fes // self.ps
        total_fes = np.arange(curr_fes, curr_fes + batch_size*self.ps, step=self.ps)  # (batch_size,)
        total_process = np.searchsorted(self.fes_thresholds, total_fes, side='right') - 1  # (batch_size,)
        old_process = np.r_[self.fes_process, total_process[:-1]]  # (batch_size,)
        check_matrix = (total_process > old_process).astype(int)
        update_matrix = np.tile(check_matrix[:, None], (1, self.ps))  # (batch_size, ps)
        pos_matrix = (self.pos + np.add.accumulate(update_matrix, axis=0)) % self.n_problem
        weight_matrix = np.zeros((batch_size, self.ps, self.n_problem))
        weight_matrix[np.arange(batch_size)[:, None], np.arange(self.ps), pos_matrix] = 1
        if change_flag:
            self.weights = weight_matrix[-1]
            self.pos = pos_matrix[-1]
            self.fes_process = total_process[-1]
        return weight_matrix










