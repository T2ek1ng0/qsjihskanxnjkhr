from metaevobox import Config, Trainer
from my_utils import construct_problem_set
from gleet_optimizer import GLEET_Optimizer
from gleet_agent import GLEET
from my_dynamic_dataset import Dynamic_Dataset

# put user-specific configuration
config = {'train_problem': 'dynamic-problem',
          'train_batch_size': 8,  # 8
          'train_parallel_mode': 'dummy',  # dummy/subproc/ray/ray-subproc
          'max_epoch': 20,  # 100
          'train_mode': 'multi',  # multi/single
          }
config = Config(config)
# construct dataset
config, datasets = construct_problem_set(config)
# initialize your MetaBBO's meta-level agent & low-level optimizer
opt = GLEET_Optimizer(config)
agent = GLEET(config)
trainer = Trainer(config, agent, opt, datasets)
if __name__ == "__main__":
    trainer.train()


