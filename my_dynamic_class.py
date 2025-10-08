import numpy as np
from my_weight import Sub_Problem_Weight
from my_noise import Gaussian_noise
from metaevobox.environment.problem.basic_problem import Basic_Problem

class Dynamic_Problem:
    def __init__(self, problem_list: list,
                 population_weight: list[Sub_Problem_Weight],
                 noise: Gaussian_noise,
                 max_fes):
        self.problem_list = problem_list
        self.population_weight = population_weight
        self.prob_noise = noise
        self.dim = []
        self.lb = []
        self.ub = []
        self.optimum = []
        self.rho = []
        self.nopt = []
        self.T1 = []
        self.fes = 0
        self.max_fes = max_fes

    def reset(self):
        self.T1 = [0] * len(self.problem_list)
        for problem in self.problem_list:
            self.dim.append(problem.dim)
            self.lb.append(problem.lb)
            self.ub.append(problem.ub)
            self.optimum.append(problem.optimum)
            self.rho.append(problem.rho)
            self.nopt.append(problem.nopt)

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        result = []
        for pop_idx, per_subpro_weight in enumerate(self.population_weight):  # 种群权重大小 (ps,n_problem)  problem.func(x)大小 (ps, 1)
            if per_subpro_weight.check(self.fes):  # 评估次数满足变化条件，切换子问题
                per_subpro_weight.weights[:] = 0
                per_subpro_weight.weights[per_subpro_weight.pos] = 1
                per_subpro_weight.pos = (per_subpro_weight.pos + 1) % len(self.problem_list)

            res = 0.0
            for idx, problem in enumerate(self.problem_list):
                res += per_subpro_weight.weights[idx] * self.prob_noise.make_noise(self.fes, self.max_fes, problem.func(x[pop_idx].reshape(1, -1))).item()
                self.fes += 1
            result.append(res)

        return result
