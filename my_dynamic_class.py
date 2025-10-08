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
        self.population_weight = population_weight  # (ps,len(problem_list))
        self.noise = noise
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
        self.fes = 0
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
        weights = self.population_weight.cal_weight(self.fes, per_cost_fes=1)  # (ps,len(problem_list))
        result = []
        for pop_idx, per_subpro_weight in enumerate(weights):
            res = 0.0
            for idx, problem in enumerate(self.problem_list):
                x_input = x[pop_idx, :problem.dim]
                res += per_subpro_weight[idx] * self.noise.make_noise(self.fes, self.max_fes,
                                                                      problem.func(x_input)).item()
                self.fes += 1
            result.append(res)
        return result

