import numpy as np
from my_weight import *
from my_noise import Gaussian_noise
from metaevobox.environment.problem.basic_problem import Basic_Problem

class Dynamic_Problem:
    def __init__(self, problem_list: list,
                 sub_problem_weight: Sub_Problem_Weight,
                 noise: Gaussian_noise):
        self.problem_list = problem_list
        self.sub_problem_weight = sub_problem_weight
        self.prob_noise = noise
        self.dim = []
        self.lb = []
        self.ub = []
        self.optimum = []
        self.rho = []
        self.nopt = []
        self.T1 = []
        self.fes = 0

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
        result = 0
        if self.sub_problem_weight.check(self.fes):  # 评估次数满足变化条件，切换子问题
            self.sub_problem_weight.weights[:] = 0
            self.sub_problem_weight.weights[self.sub_problem_weight.pos] = 1
            self.sub_problem_weight.pos = (self.sub_problem_weight.pos + 1) % len(self.problem_list)

        for idx, problem in enumerate(self.problem_list):
            result += self.sub_problem_weight.weights[idx] * self.prob_noise.make_noise(problem.FES, problem.maxfes, problem.eval(x))
            self.fes += x.shape[0]
        return result
