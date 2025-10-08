import numpy as np
from collections import deque
from typing import Union

class Sub_Problem_Weight:
    def __init__(self, n_problem, weights, max_fes, rates: Union[list, np.ndarray, float]):
        if isinstance(rates, list):
            rates = np.array(rates, dtype=float)
        self.rates = rates
        self.n_problem = n_problem
        self.weights = np.array(weights, dtype=float)
        self.max_fes = max_fes

        if 1 in self.weights:
            self.pos = int(np.where(self.weights == 1)[0][0])  # pos 指向当前子问题
        else:
            self.pos = 0
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


