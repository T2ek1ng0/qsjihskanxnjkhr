import numpy as np
from my_weight import Sub_Problem_Weight
from my_noise import Gaussian_noise
from metaevobox.environment.problem.basic_problem import Basic_Problem

class Dynamic_Problem:
    def __init__(self, problem_list: list[Basic_Problem],
                 population_weight: Sub_Problem_Weight,
                 noise: Gaussian_noise,
                 max_fes):
        self.problem_list = problem_list
        self.n_problem = len(self.problem_list)
        self.ps = None
        self.population_weight = population_weight
        self.noise = noise
        self.dim = 0
        self.optimum = []
        self.T1 = [0] * self.n_problem
        self.fes = 0
        self.maxfes = max_fes
        self.ub = problem_list[self.population_weight.pos].ub
        self.lb = problem_list[self.population_weight.pos].lb

    def __str__(self):
        return "Dynamic Problem"

    def reset(self):
        self.T1 = [0] * self.n_problem
        self.fes = 0
        self.ub = self.problem_list[self.population_weight.pos].ub
        self.lb = self.problem_list[self.population_weight.pos].lb
        for problem in self.problem_list:
            self.optimum.append(problem.optimum)
            self.dim = max(self.dim, problem.dim)

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.ps = x.shape[0]
        noise = self.noise.make_noise(self.fes, self.maxfes, self.ps, self.ps)  # (ps,1)
        weights = self.population_weight.cal_weight(self.ps, self.fes, self.ps, change_flag=True)  # (ps,n_problem)
        self.ub = self.problem_list[self.population_weight.pos].ub
        self.lb = self.problem_list[self.population_weight.pos].lb
        fitness = np.zeros((self.ps, self.n_problem))
        for prob_idx, problem in enumerate(self.problem_list):
            x_input = x[:, :problem.dim]
            x_input = np.clip(x_input, problem.lb, problem.ub)
            fitness[:, prob_idx] = problem.func(x_input)
        self.fes += self.ps
        result = np.sum(weights * fitness, axis=1)
        return result + noise

    def re_eval(self, x, mode='noise'):  # (self.gen*5, dim)
        n_archive = x.shape[0]
        weights = self.population_weight.re_eval_weight(self.fes, n_archive)
        noise = self.noise.re_eval_noise(self.fes, self.maxfes, n_archive)
        fitness = np.zeros((n_archive, self.n_problem))
        for prob_idx, problem in enumerate(self.problem_list):
            x_input = x[:, :problem.dim]
            x_input = np.clip(x_input, problem.lb, problem.ub)
            fitness[:, prob_idx] = problem.func(x_input)
        result = np.sum(weights * fitness, axis=1)
        if mode == 'noise':
            self.fes += n_archive
            return result + noise
        else:
            return result






