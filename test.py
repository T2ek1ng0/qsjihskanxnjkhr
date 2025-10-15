import numpy as np
from my_dynamic_class import Dynamic_Problem
from my_weight import Sub_Problem_Weight
from my_noise import Gaussian_noise
from metaevobox import Config, Trainer
from metaevobox.environment.problem.utils import construct_problem_set
from gleet_optimizer import GLEET_Optimizer
from gleet_agent import GLEET

# put user-specific configuration
config = {'train_problem': 'bbob-10D',
          'train_batch_size': 2,
          'train_parallel_mode': 'dummy',  # dummy/subproc/ray/ray-subproc
          'max_learning_step': 1000,
          }
config = Config(config)
np.random.seed(config.seed)
# construct dataset
config, datasets = construct_problem_set(config)
opt = GLEET_Optimizer(config)
train_problem_set = datasets[0]
train_problem_set = [p[0] for p in train_problem_set]
test_problem_set = datasets[1]
test_problem_set = [p[0] for p in test_problem_set]
max_fes = config.maxFEs
my_noise = Gaussian_noise(begin_std=0.0, end_std=10.0)
train_population_weight = Sub_Problem_Weight(len(train_problem_set), opt.ps, max_fes, rates=[0.04, 0.12, 0.16, 0.24])
train_problem = Dynamic_Problem(train_problem_set, train_population_weight, my_noise, max_fes)
test_population_weight = Sub_Problem_Weight(len(test_problem_set), opt.ps, max_fes, rates=[0.04, 0.12, 0.16, 0.24])
test_problem = Dynamic_Problem(test_problem_set, test_population_weight, my_noise, max_fes)
gleet = GLEET(config)
train_problem.reset()
opt.init_population(train_problem)
action = np.tile([5, 2.05, 2.05], (opt.ps, 1))
for _ in range(100):
    opt.update(action, train_problem)
print(opt.archive_newval)
print(len(opt.archive_newval))


