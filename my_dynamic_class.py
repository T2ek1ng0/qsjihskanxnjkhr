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
        self.ps = population_weight.get_weight().shape[0]
        self.population_weight = population_weight  # (ps,n_problem)
        self.noise = noise
        self.dim = []
        self.lb = []
        self.ub = []
        self.optimum = []
        self.T1 = []
        self.fes = [0] * self.ps
        self.maxfes = max_fes

    def reset(self):
        self.T1 = [0] * self.n_problem
        self.fes = [0] * self.ps
        for problem in self.problem_list:
            self.dim.append(problem.dim)
            self.lb.append(problem.lb)
            self.ub.append(problem.ub)
            self.optimum.append(problem.optimum)

    def eval(self, x):
        if not isinstance(x, np.ndarray):
            x = np.array(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        weights = self.population_weight.cal_weight(self.fes, per_cost_fes=1)  # (ps,n_problem)
        noise = self.noise.make_noise(self.fes, self.maxfes, self.n_problem)  # (ps,n_problem)
        result = []
        for pop_idx, per_subpro_weight in enumerate(weights):  # 遍历种群中的每个个体
            res = 0.0
            for prob_idx, problem in enumerate(self.problem_list):  # 遍历问题列表
                x_input = x[pop_idx, :problem.dim].reshape(1, -1)
                res += per_subpro_weight[prob_idx] * (problem.func(x_input) + noise[pop_idx][prob_idx])
            result.append(res.item())  # (1,1)的 np.ndarray 转回 float
            self.fes[pop_idx] += 1
        return result

