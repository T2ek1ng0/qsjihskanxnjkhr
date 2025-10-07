import numpy as np
from collections import deque

class population_weight:
    def __init__(self, weights, rates, max_fes):
        self.weights = weights
        self.queue = deque((rates[i]*max_fes, rates[i+1]*max_fes) for i in range(len(rates)-1))  # rates 像 [0.2,0.4,0.6,0.8] 这样

    def check(self, current_fes):
        while self.queue and current_fes > self.queue[0][1]:
            self.queue.popleft()

        if self.queue:
            lb, ub = self.queue[0]
            if lb <= current_fes <= ub:
                self.queue.popleft()
                return True
        return False

    def change(self, problem, curr_fes, curr_x, fes_cost=None):
        if fes_cost is None:
            fes_cost = curr_x.shape[0]
        if self.check(curr_fes + fes_cost):
            problem.eval(curr_x)

class sub_problem_weight:
    def __init__(self, n_problem, weights, rates, max_fes):
        self.queue = deque((rates[i] * max_fes, rates[i + 1] * max_fes, i) for i in range(len(rates) - 1))  # rates 像 [0.2,0.4,0.6,0.8] 这样
        self.rates = rates
        self.n_problem = n_problem
        self.weights = weights

    def check(self, current_fes):
        while self.queue and current_fes > self.queue[0][1]:
            self.queue.popleft()

        if self.queue:
            lb, ub, idx = self.queue[0]
            if lb <= current_fes <= ub:
                self.queue.popleft()
                return idx
        return -1

    def change(self, curr_fes, curr_x, fes_cost=None):
        if fes_cost is None:
            fes_cost = curr_x.shape[0]
        idx = self.check(curr_fes + fes_cost)
        if idx != -1:
            new_weights = np.zeros(self.n_problem)
            new_weights[idx] = 1
            self.weights = new_weights
        return self.weights

