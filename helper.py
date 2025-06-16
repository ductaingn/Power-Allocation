from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.logger import Logger
from wandb.integration.sb3 import WandbCallback
from typing import Optional, Literal
import wandb
import pandas as pd

class WandbLoggingCallback(WandbCallback):
    def __init__(        
        self,
        logger:Logger,
        verbose: int = 0,
        model_save_path: Optional[str] = None,
        model_save_freq: int = 0,
        gradient_save_freq: int = 1,
        log: Optional[Literal["gradients", "parameters", "all"]] = "all"):
        super().__init__(
            verbose=verbose,
            model_save_path=model_save_path,
            model_save_freq=model_save_freq,
            gradient_save_freq=gradient_save_freq,
            log="all"
        )
        self.custom_logger = logger

    def _pre_process_log(self, info:dict) -> dict:
        # Pre-process the info dictionary to remove unwanted keys
        # This is just an example, you can modify it as per your requirements
        keys_to_remove = ["terminal_observation", " TimeLimit.truncated", "episode.t", "episode.l", "episode.r"]
        for key in keys_to_remove:
            info.pop(key, None)
        
        info["Loss/ Actor Loss"] = self.custom_logger.name_to_value["train/actor_loss"]
        info["Loss/ Critic Loss"] = self.custom_logger.name_to_value["train/critic_loss"]
        info["Loss/ Entropy Loss"] = self.custom_logger.name_to_value["train/entropy_loss"]
        info["Overall/ Learning Rate"] = self.custom_logger.name_to_value["train/learning_rate"]
        return info

    def on_step(self) -> bool:
        # 'infos' is a list of info dicts for each environment in the vecenv
        infos = self.locals.get("infos", [])
        for info in infos:
            if info:  # Log only if info is not empty
                log = self._pre_process_log(info)
                wandb.log(log, commit=True)
        return True
    
def _pre_process_log(info:dict) -> dict:
    # Pre-process the info dictionary to remove unwanted keys
    # This is just an example, you can modify it as per your requirements
    keys_to_remove = ["terminal_observation", " TimeLimit.truncated", "episode.t", "episode.l", "episode.r"]
    for key in keys_to_remove:
        info.pop(key, None)

    return info
    
def custom_callback(locals, globals):
    """
    Callback to log for Random scenario
    """
    infos = locals.get("infos", [])
    for info in infos:
        if info:  # Log only if info is not empty
            log = _pre_process_log(info)
            wandb.log(log, commit=True)
    return True

def get_log_from_wandb(run_id:str, run_name:str=None, entity:str="ductaingn-015203-none", project="PowerAllocation", return_run:bool=False) -> pd.DataFrame:
    api = wandb.Api()
    if run_name:
        run = api.runs(f"{entity}/{project}", filters={"displayName": run_name})
        if len(run) == 0:
            raise ValueError(f"Run with name {run_name} not found in project {project} for entity {entity}.")
        if len(run) > 1:
            raise ValueError(f"Multiple runs found with name {run_name} in project {project} for entity {entity}. Please specify a unique run name.")

        run_id = run[0].id

    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    records = []
    for row in run.scan_history():  # no keys passed = get all keys
        records.append(row)
    history = pd.DataFrame(records)
    if run.config['algorithm'] == "Random":
        history['Algorithm'] = "Random"
    elif run.config['algorithm'] == "RAQL":
        history['Algorithm'] = "RAQL"
    elif run.config['algorithm'] == "LearnInterface":
        history['Algorithm'] = "SACPF" 
    elif run.config['algorithm'] == "LearnInterfaceAndPower":
        history['Algorithm'] = "SACPA" 
    
    if return_run:
        return history, run
    else:
        return history