from metaevobox.environment.problem.utils import construct_problem_set
from metaevobox import Config, get_baseline
from my_tester import Tester
from gleet_optimizer import GLEET_Optimizer
from fixedact_nbnc_optimizer import basic_nbnc_Optimizer
from my_dynamic_dataset import Dynamic_Dataset

# specify your configuration
config = {
    'train_problem': 'bbob-10D',
    'train_batch_size': 2,
    'train_parallel_mode': 'dummy',  # dummy/subproc/ray/ray-subproc
    'max_learning_step': 1000,
    'max_epoch': 10,
    'test_problem': 'bbob-10D',  # specify the problem set you want to benchmark
    'test_batch_size': 2,
    'test_difficulty': 'easy',  # this is a train-test split mode
    'test_parallel_mode': 'Serial',  # 'Full', 'Baseline_Problem', 'Problem_Testrun', 'Batch', 'Serial'
    'baselines': {
        # your MetaBBO
        'NBNC': {
            'agent': 'GLEET',
            'optimizer': GLEET_Optimizer,
            'model_load_path': r"agent_model\train\GLEET\20251018T011720_bbob-10D_easy\checkpoint-10.pkl",
        },

        # Other baselines to compare
        'Fixed_parameter': {'optimizer': basic_nbnc_Optimizer},
    },
}
config = Config(config)
# construct dataset
config, datasets = construct_problem_set(config)
dynamic_datasets = Dynamic_Dataset.get_dataset(config, datasets)
# initialize all baselines to compare (yours + others)
baselines, config = get_baseline(config)
# initialize tester
tester = Tester(config, baselines, dynamic_datasets)
# test
if __name__ == '__main__':
    tester.test()
