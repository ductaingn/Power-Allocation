import random
import numpy as np
import torch
import pandas as pd
import optuna
from optuna.visualization import (
    plot_pareto_front,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    plot_slice
)
from train import Trainer
from copy import deepcopy
import os
import argparse
import yaml
from functools import partial

parser = argparse.ArgumentParser(description="Tune hyperparams for algorithms in RAORESMIN problem")
parser.add_argument('-b', '--base_path', type=str, required=True, help='Base path for configs and data')
parser.add_argument('-n', '--num_runs', type=int, default=3, help='Number of runs for each configuration')
parser.add_argument('-s', '--scenarios', type=int, nargs='+', default=[1, 2], help='List of scenarios to run')
parser.add_argument('-p', '--powers', type=int, nargs='+', default=[5, 1], help='List of power levels in dBm to run')
parser.add_argument('-a', '--algorithms', type=str, nargs='+', default=["SACPA"], help='List of algorithms to run')
args = parser.parse_args()

def objective(trial:optuna.Trial, **kwargs) -> float:
    """
    Objective function for hyperparameter tuning using Optuna.
    
    :param trial: An Optuna trial object.
    
    :return: The value of the objective function to minimize.
    """
    ...
    default_train_configs:dict = yaml.safe_load(
        open(os.path.join(kwargs['base_path'], "train_config.yaml"))
    )
    print(f"Loaded training configurations: {default_train_configs}")

    seed = default_train_configs.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.01)
    reward_qos_coef = trial.suggest_float("reward_qos_coef", 0.1, 10.0)
    reward_power_coef = trial.suggest_float("reward_power_coef", 5.0, 20.0)

    trial_configs = deepcopy(default_train_configs)
    trial_configs['sac_hyperparams'].update({
        'learning_rate': learning_rate,
        'gamma': gamma,
        'tau': tau
    })

    trial_configs['env_config']['reward_coef'] = {
        'reward_qos': reward_qos_coef,
        'reward_power': reward_power_coef
    }

    run_name = f"trial_{trial.number}"

    try:
        trainer = Trainer(train_configs=trial_configs)
        results:pd.DataFrame = trainer.train(run_name=run_name, tune=True).transpose()

        psr = results['Avg. Success']
        reward = results['Reward']

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return (-float('inf'), -float('inf'))
    
    return psr, reward

if __name__ == "__main__":
    BASE_PATH = args.base_path
    if not os.path.exists(BASE_PATH):
        raise FileNotFoundError(f"Base path {BASE_PATH} does not exist.")
    print(f"Base path: {BASE_PATH}")

    num_runs = args.num_runs
    scenarios = args.scenarios
    powers = args.powers
    algorithms = args.algorithms

    print(f"Tuning hyperparams with {num_runs} runs, scenarios: {scenarios}, power levels: {powers}, algorithms: {algorithms}")

    kwargs = {
        'base_path': BASE_PATH,
        'num_runs': num_runs,
        'scenarios': scenarios,
        'powers': powers,
        'algorithms': algorithms
    }

    study = optuna.create_study(directions=['maximize', 'maximize'],)
    study.optimize(
        func=partial(objective, **kwargs),
        n_trials=5,
        show_progress_bar=True
    )

    pareto_front = study.best_trials
    df = pd.DataFrame([t.params | dict(zip(["psr", "reward"], t.values)) for t in pareto_front])
    print(f"Best trials: {df}")
    try:
        df.to_csv(os.path.join(BASE_PATH, "pareto_front.csv"), index=False)
    except Exception as e:
        print(f"Error saving pareto front data: {e}")

    # fig = plot_pareto_front(study, target_names=["Avg. Success", "Reward"])
    # fig.show()

    # fig1 = plot_param_importances(study, target=lambda t: t.values[0], target_name="Avg. Success")
    # fig2 = plot_param_importances(study, target=lambda t: t.values[1], target_name="Reward")

    # fig1.show()
    # fig2.show()