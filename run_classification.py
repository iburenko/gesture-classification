import os
from argparse import ArgumentParser
import logging
from time import localtime, strftime

import hydra
from omegaconf import OmegaConf, DictConfig

from gesture_classification.trainer import trainer, foo


slurm_job_id = os.getenv("SLURM_JOB_ID")
start_time = strftime("%d_%m_%Y_%H:%M:%S", localtime())

logger = logging.getLogger(__name__)

@hydra.main(config_path="conf", config_name="base_config")
def main(cfg: DictConfig) -> None:    
    # logger_name = cfg.logging.log_name

    # log_folder = f"id_{slurm_job_id}_{start_time}_{logger_name}"
    # ex = submitit.AutoExecutor(log_folder)
    # ex.update_parameters(
    #     timeout_min=240,
    #     gres="gpu:a40:1"
    # )
    # job = ex.submit(foo, cfg)
    # logger.info(job.job_id)
    # output = job.result()
    # logger.info(output)
    trainer(cfg)


if __name__ == "__main__":
    main()
