import numpy as np
from typing import Union
from my_weight import Sub_Problem_Weight
from my_noise import Gaussian_noise
from metaevobox.environment.problem.basic_problem import Basic_Problem

class Dynamic_Problem:
    def __init__(self, max_fes: int,
                 problem_list: Union[list[Basic_Problem], Basic_Problem],
                 population_weight: Union[Sub_Problem_Weight, None] = None,
                 noise: Union[Gaussian_noise, None] = None):
        self.problem_list = problem_list
        self.n_problem = len(self.problem_list) if isinstance(problem_list, list) else 1
        self.population_weight = population_weight
        self.noise = noise
        self.dim = 0
        self.optimum = []
        self.T1 = [0] * self.n_problem
        self.fes = 0
        self.maxfes = max_fes
        if self.population_weight:
            self.ub = self.problem_list[self.population_weight.pos].ub
            self.lb = self.problem_list[self.population_weight.pos].lb
            self.optimum = [problem.optimum for problem in self.problem_list]
            for problem in self.problem_list:
                self.dim = max(self.dim, problem.dim)
        else:
            self.ub = self.problem_list.ub
            self.lb = self.problem_list.lb
            self.optimum = self.problem_list.optimum
            self.dim = self.problem_list.dim

    def __str__(self):
        return "Dynamic Problem"

    def reset(self):
        self.T1 = [0] * self.n_problem
        self.fes = 0
        if self.population_weight:
            self.ub = self.problem_list[self.population_weight.pos].ub
            self.lb = self.problem_list[self.population_weight.pos].lb
            self.optimum = [problem.optimum for problem in self.problem_list]
            for problem in self.problem_list:
                self.dim = max(self.dim, problem.dim)
        else:
            self.ub = self.problem_list.ub
            self.lb = self.problem_list.lb
            self.optimum = self.problem_list.optimum
            self.dim = self.problem_list.dim


    def eval(self, x, mode='noise'):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        ps = x.shape[0]
        noise = self.noise.make_noise(self.fes, self.maxfes, ps, ps) if self.noise is not None else 0  # (ps,1)
        if self.population_weight:
            weights = self.population_weight.cal_weight(ps, self.fes, ps, change_flag=True)  # (ps,n_problem)
            self.ub = self.problem_list[self.population_weight.pos].ub
            self.lb = self.problem_list[self.population_weight.pos].lb
            self.optimum = self.problem_list[self.population_weight.pos].optimum
            fitness = np.stack([problem.func(np.clip(x[:, :problem.dim], problem.lb, problem.ub)) for problem in self.problem_list],axis=1)  # (ps,n_problem)
            result = np.sum(weights * fitness, axis=1)
        else:
            result = self.problem_list.func(np.clip(x[:, :self.dim], self.lb, self.ub))
        if mode == 'noise':
            self.fes += ps
            return result + noise
        elif mode == 'real':
            return result






