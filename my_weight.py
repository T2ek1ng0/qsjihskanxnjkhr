import numpy as np
from collections import deque
from typing import Union

class Population_Weight:
    def __init__(self, weights: list, rates: Union[list, float], max_fes):
        self.weights = weights
        self.rates = rates
        self.max_fes = max_fes
        if isinstance(self.rates, list):
            self.queue = deque((rates[i]*max_fes, rates[i+1]*max_fes) for i in range(len(rates)-1))  # rates 像 [0.2,0.4,0.6,0.8] 这样
        else:
            self.threshold = rates * max_fes

    def check(self, current_fes):
        if isinstance(self.rates, list):
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

class Sub_Problem_Weight:
    def __init__(self, n_problem, weights: list, rates: Union[list, float], max_fes):
        self.rates = rates
        self.n_problem = n_problem
        self.weights = weights
        self.max_fes = max_fes
        if 1 in self.weights:
            self.pos = self.weights.index(1)
        else:
            self.pos = 0
        if isinstance(self.rates, list):
            self.queue = deque((rates[i]*max_fes, rates[i+1]*max_fes) for i in range(len(rates)-1))  # rates 像 [0.2,0.4,0.6,0.8] 这样
        else:
            self.threshold = rates * max_fes

    def check(self, current_fes):
        if isinstance(self.rates, list):
            while self.queue and current_fes > self.queue[0][1]:
                self.queue.popleft()

            if self.queue:
                lb, ub= self.queue[0]
                if lb <= current_fes <= ub:
                    self.queue.popleft()
                    return True
            return False
        else:
            if current_fes >= self.threshold:
                self.threshold += self.max_fes * self.rates
                return True
            return False

    def change(self, curr_fes, curr_x, fes_cost=None):
        if fes_cost is None:
            fes_cost = curr_x.shape[0]
        if self.check(curr_fes + fes_cost):
            self.weights[:] = 0
            self.weights[self.pos] = 1
            self.pos = (self.pos + 1) % self.n_problem
        return self.weights


