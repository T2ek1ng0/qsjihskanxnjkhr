from metaevobox import Config, Trainer
from metaevobox.environment.problem.utils import construct_problem_set
from gleet_optimizer import GLEET_Optimizer
from gleet_agent import GLEET
from my_dynamic_dataset import Dynamic_Dataset

# put user-specific configuration
config = {'train_problem': 'bbob-10D',
          'train_batch_size': 2,
          'train_parallel_mode': 'dummy',  # dummy/subproc/ray/ray-subproc
          'max_learning_step': 1000,
          'max_epoch': 10,
          }
config = Config(config)
# construct dataset
config, datasets = construct_problem_set(config)
dynamic_datasets = Dynamic_Dataset.get_dataset(config, datasets)
# initialize your MetaBBO's meta-level agent & low-level optimizer
opt = GLEET_Optimizer(config)
agent = GLEET(config)
trainer = Trainer(config, agent, opt, dynamic_datasets)
if __name__ == "__main__":
    trainer.train()


