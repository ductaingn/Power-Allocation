import random
import numpy as np
import torch
import yaml
from train import Trainer
import os
import argparse

parser = argparse.ArgumentParser(description="Benchmark algorithms in RAORESMIN problem")
parser.add_argument('-b', '--base_path', type=str, required=True, help='Base path for configs and data')
parser.add_argument('-n', '--num_runs', type=int, default=3, help='Number of runs for each configuration')
parser.add_argument('-s','--scenarios', type=int, nargs='+', default=[1, 2], help='List of scenarios to run')
parser.add_argument('-p','--powers', type=int, nargs='+', default=[5, 1], help='List of power levels in dBm to run')
parser.add_argument('-a','--algorithms', type=str, nargs='+', default=["Random", "RAQL", "SACPF", "SACPA"], help='List of algorithms to run')
args = parser.parse_args()

if __name__ == "__main__":
    BASE_PATH = args.base_path
    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"Base path {BASE_PATH} does not exist.")
    print(f"Base path: {BASE_PATH}")

    num_runs = args.num_runs
    scenarios = args.scenarios
    powers = args.powers
    algorithms = args.algorithms
    print(f"Running benchmark with {num_runs} runs, scenarios: {scenarios}, power levels: {powers}, algorithms: {algorithms}")

    train_configs:dict = yaml.safe_load(
        open(os.path.join(BASE_PATH, "train_config.yaml"))
    )
    print(f"Loaded training configurations: {train_configs}")

    seed = train_configs.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    for run in range(num_runs):
        for scenario in scenarios:
            for power in powers:
                for algorithm in algorithms:
                    P_sum = pow(10, power/10)*1e-3
                    
                    train_configs['env_config']['P_sum'] = P_sum
                    train_configs['env_config']['algorithm'] = algorithm
                    train_configs['env_config']['h_tilde_path'] = os.path.join(BASE_PATH, f'environment/data/scenario_{scenario}/h_tilde.pickle')
                    train_configs['env_config']['devices_positions_path'] = os.path.join(BASE_PATH, f'environment/data/scenario_{scenario}/device_positions.pickle')
                    train_configs['env_config']['num_devices'] = 10 if scenario==1 else 15

                    trainer = Trainer(train_configs)

                    trainer.train(run_name=f'{algorithm}_scenario{scenario}_{power}dbm_{run+1}')