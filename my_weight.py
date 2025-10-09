import numpy as np
from collections import deque
from typing import Union

class Sub_Problem_Weight:
    def __init__(self, n_problem, weights, max_fes, rates: Union[list, np.ndarray, float]):
        if isinstance(rates, list):
            rates = np.array(rates, dtype=float)
        elif isinstance(rates, float):
            if not (0 < rates <= 1.0):
                raise ValueError(f"Invalid rate value {rates}, must be in (0, 1].")
            temp_rates = np.arange(start=rates, stop=1.0, step=rates)
            rates = np.round(temp_rates, 4)
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
        self.pos = []  # 指向每个粒子当前的子问题索引，默认为 0

        self.pos = np.argmax(self.weights, axis=1)

        self.total_queue = [deque((rates[i] * max_fes, rates[i + 1] * max_fes) for i in range(len(rates) - 1)) for _ in range(self.ps)]

    def check(self, current_fes, pop_idx):  # 判定当前个体的评估次数是否满足权重变化要求
        while self.total_queue[pop_idx] and current_fes > self.total_queue[pop_idx][0][1]:
            self.total_queue[pop_idx].popleft()

        if self.total_queue[pop_idx]:
            lb, ub = self.total_queue[pop_idx][0]
            if lb <= current_fes <= ub:
                self.total_queue[pop_idx].popleft()
                return True
        return False

    def cal_weight(self, total_fes: list, per_cost_fes):  # total_fes: ps, weights: (ps,n_problem)
        for pop_idx, per_fes in enumerate(total_fes):
            if self.check(per_fes, pop_idx):
                self.weights[pop_idx][:] = 0
                self.pos[pop_idx] = (self.pos[pop_idx] + 1) % self.n_problem
                self.weights[pop_idx][self.pos[pop_idx]] = 1
        return self.weights

    def get_weight(self):
        return self.weights


