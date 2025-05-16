import random
import numpy as np
import torch
import yaml
from train import Trainer

if __name__ == "__main__":
    train_configs:dict = yaml.safe_load(open("train_config.yaml"))
    seed = train_configs.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_run = 2

    for runtime in range(num_run):
        for power in range(1,5):
            for algorithm in ["Random", "LearnInterfaceAndPower", "LearnInterface"]:
                P_sum = pow(10, power/10)*1e-3
                
                train_configs['env_config']['P_sum'] = P_sum
                train_configs['env_config']['algorithm'] = algorithm

                trainer = Trainer(train_configs)

                trainer.train(run_name=f'{algorithm}_{power}dbm_{runtime}')