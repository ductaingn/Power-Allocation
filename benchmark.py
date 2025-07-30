import random
import numpy as np
import torch
import yaml
from train import Trainer
import os
import argparse

parser = argparse.ArgumentParser(description="Benchmark Algorithms in RAORESMIN problem")
parser.add_argument('--base_path', type=str, required=True, help='Base path for configs and data')
parser.add_argument('--num_runs', type=int, default=3, help='Number of runs for each configuration')
parser.add_argument('--scenarios', type=int, nargs='+', default=[1, 2], help='List of scenarios to run')
parser.add_argument('--algorithms', type=str, nargs='+', default=["Random", "RAQL", "SACPF", "SACPA"], help='List of algorithms to run')
args = parser.parse_args()

if __name__ == "__main__":
    BASE_PATH = args.base_path
    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"Base path {BASE_PATH} does not exist.")
    print(f"Base path: {BASE_PATH}")
    num_runs = args.num_runs
    scenarios = args.scenarios
    algorithms = args.algorithms
    print(f"Running benchmark with {num_runs} runs, scenarios: {scenarios}, algorithms: {algorithms}")

    train_configs:dict = yaml.safe_load(
        open(os.path.join(BASE_PATH, "train_config.yaml"))
    )
    print(f"Loaded training configurations: {train_configs}")

    seed = train_configs.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for runtime in range(num_runs):
        for scenario in [1, 2]:
            for power in [1, 5]:
                for algorithm in ["Random", "SACPF", "RAQL", "SACPA"]:
                    P_sum = pow(10, power/10)*1e-3
                    
                    train_configs['env_config']['P_sum'] = P_sum
                    train_configs['env_config']['algorithm'] = algorithm
                    train_configs['env_config']['h_tilde_path'] = os.path.join(BASE_PATH, f'environment/data/scenario_{scenario}/h_tilde.pickle')
                    train_configs['env_config']['devices_positions_path'] = os.path.join(BASE_PATH, f'environment/data/scenario_{scenario}/device_positions.pickle')
                    train_configs['env_config']['num_devices'] = 10 if scenario==1 else 15

                    trainer = Trainer(train_configs)

                    trainer.train(run_name=f'{algorithm}_scenario{scenario}_{power}dbm_{runtime}')