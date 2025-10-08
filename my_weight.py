import numpy as np
from collections import deque
from typing import Union

class Sub_Problem_Weight:
    def __init__(self, n_problem, weights, max_fes, rates: Union[list, np.ndarray, float]):
        if isinstance(rates, list):
            rates = np.array(rates, dtype=float)
        self.rates = rates
        self.n_problem = n_problem
        self.weights = np.array(weights, dtype=float)  # (ps,n_problem)
        self.max_fes = max_fes
        self.pos = []  # 指向每个粒子当前的子问题，默认为 0

        for per_weight in self.weights:
            if 1 in per_weight:
                self.pos.append(int(np.where(per_weight == 1)[0][0]))
            else:
                self.pos.append(0)

        if isinstance(self.rates, np.ndarray):
            self.queue = deque((rates[i]*max_fes, rates[i+1]*max_fes) for i in range(len(rates)-1))  # rates 像 [0.2,0.4,0.6,0.8] 这样
        else:
            self.threshold = rates * max_fes

    def check(self, current_fes):
        if isinstance(self.rates, np.ndarray):
            while self.queue and current_fes > self.queue[0][1]:
                self.queue.popleft()

            if self.queue:
                lb, ub = self.queue[0]
                if lb <= current_fes <= ub:
                    self.queue.popleft()
                    return True
            return False
        else:
            if current_fes >= self.threshold:
                self.threshold += self.max_fes * self.rates
                return True
            return False

    def cal_weight(self, curr_fes, per_cost_fes):
        for idx, per_weight in enumerate(self.weights):
            if self.check(curr_fes):
                self.weights[idx][:] = 0
                self.pos[idx] = (self.pos[idx] + 1) % self.n_problem
                self.weights[idx][self.pos[idx]] = 1
            curr_fes += per_cost_fes
        return self.weights

    def get_weight(self):
        return self.weights


