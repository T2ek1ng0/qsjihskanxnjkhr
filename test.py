import numpy as np
from my_dynamic_class import Dynamic_Problem
from my_weight import Sub_Problem_Weight
from my_noise import Gaussian_noise
from metaevobox import Config, Trainer
from metaevobox.environment.problem.utils import construct_problem_set

# put user-specific configuration
config = {'train_problem': 'bbob-10D',
          'train_batch_size': 2,
          'train_parallel_mode': 'dummy',  # dummy/subproc/ray/ray-subproc
          }
config = Config(config)
np.random.seed(config.seed)
# construct dataset
config, datasets = construct_problem_set(config)
test_problem_set = datasets[1]
test_problem_set = [p[0] for p in test_problem_set]
n_problem = len(test_problem_set)
max_fes = 100
my_noise = Gaussian_noise(begin_std=0.0, end_std=10.0)
ps = 4
population_weight = []
random_x = np.random.rand(ps, 10)
for _ in range(ps):
    random_subprob_weight = np.zeros(n_problem)
    random_index = np.random.randint(0, n_problem)
    random_subprob_weight[random_index] = 1
    population_weight.append(random_subprob_weight)
population_weight = Sub_Problem_Weight(n_problem, population_weight, max_fes, rates=0.02)
problem_set = Dynamic_Problem(test_problem_set, population_weight, my_noise, max_fes)

print(problem_set.eval(random_x))
print(problem_set.population_weight.get_weight())
print(problem_set.eval(random_x))
print(problem_set.population_weight.get_weight())




