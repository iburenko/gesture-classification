import os
from argparse import ArgumentParser
import logging
from time import localtime, strftime

import hydra
from omegaconf import OmegaConf, DictConfig

from gesture_classification.trainer import trainer


slurm_job_id = os.getenv("SLURM_JOB_ID")
start_time = strftime("%d_%m_%Y_%H:%M:%S", localtime())

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:    
    print("start")
    trainer(cfg)


if __name__ == "__main__":
    main()
