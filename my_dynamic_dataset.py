import numpy as np
import torch
from my_weight import Sub_Problem_Weight
from my_noise import Gaussian_noise
from my_dynamic_class import *
from basic_problem import *
from torch.utils.data import Dataset

class Dynamic_Dataset(Dataset):
    def __init__(self, data, batch_size=1):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.N = len(self.data)
        self.ptr = [i for i in range(0, self.N, batch_size)]
        self.index = np.arange(self.N)
        self.maxdim = 0
        for item in self.data:
            self.maxdim = max(self.maxdim, item.dim)

    @staticmethod
    def get_dataset(config,
                    datasets,
                    version='numpy',
                    train_batch_size=1,
                    test_batch_size=1,
                    difficulty=None,
                    user_train_list=None,
                    user_test_list=None,
                    instance_seed=3849):

        if instance_seed > 0:
            np.random.seed(instance_seed)
            torch.manual_seed(instance_seed)

        instance_class_list = ["Sphere", "Schwefel12", "Ellipsoidal", "Ellipsoidal_high_cond", "Bent_cigar",
                         "Discus", "Dif_powers", "Rosenbrock", "Ackley", "Weierstrass", "Griewank", "Rastrigin",
                         "Buche_Rastrigin", "Mod_Schwefel", "Katsuura", "Grie_rosen", "Escaffer6", "Happycat",
                         "Hgbat", "Lunacek_bi_Rastrigin", "Zakharov", "Levy", "Scaffer_F7", "Step_Rastrigin",
                         "Linear_Slope", "Attractive_Sector", "Step_Ellipsoidal", "Sharp_Ridge", "Rastrigin_F15",
                         "Schwefel", "Gallagher101", "Gallagher21"]

        train_problem_set = datasets[0]
        train_problem_set = [p[0] for p in train_problem_set]
        test_problem_set = datasets[1]
        test_problem_set = [p[0] for p in test_problem_set]
        max_fes = config.maxFEs
        my_noise = Gaussian_noise(begin_std=1.0, end_std=10.0)
        train_population_weight = Sub_Problem_Weight(len(train_problem_set), max_fes, rates=1 / len(train_problem_set))
        train_problem = Dynamic_Problem(max_fes, train_problem_set, train_population_weight, my_noise)
        test_population_weight = Sub_Problem_Weight(len(test_problem_set), max_fes, rates=1 / len(test_problem_set))
        test_problem = Dynamic_Problem(max_fes, test_problem_set, test_population_weight, my_noise)
        train_set = [train_problem]
        test_set = [test_problem]
        return Dynamic_Dataset(train_set, train_batch_size), Dynamic_Dataset(test_set, test_batch_size)

    def __getitem__(self, item):
        ptr = self.ptr[item]
        index = self.index[ptr: min(ptr + self.batch_size, self.N)]
        res = []
        for i in range(len(index)):
            res.append(self.data[index[i]])
        return res

    def __len__(self):
        return self.N

    def __add__(self, other: 'Dynamic_Dataset'):
        return Dynamic_Dataset(self.data + other.data, self.batch_size)

    def shuffle(self):
        self.index = np.random.permutation(self.N)


