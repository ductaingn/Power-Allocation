from stable_baselines3.common.callbacks import BaseCallback, EveryNTimesteps
from stable_baselines3.common.logger import Logger
from wandb.integration.sb3 import WandbCallback
from typing import Optional, Literal
import wandb

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
        
        # info["Loss/ Actor Loss"] = self.custom_logger
        info["Loss/ Actor Loss"] = self.custom_logger.name_to_value["train/actor_loss"]
        info["Loss/ Critic Loss"] = self.custom_logger.name_to_value["train/critic_loss"]
        return info

    def on_step(self) -> bool:
        # 'infos' is a list of info dicts for each environment in the vecenv
        infos = self.locals.get("infos", [])
        for info in infos:
            if info:  # Log only if info is not empty
                log = self._pre_process_log(info)
                wandb.log(log, commit=True)
        return True