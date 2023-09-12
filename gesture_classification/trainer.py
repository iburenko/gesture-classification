import os
import logging
from time import localtime, strftime

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from gesture_classification.datasets import SnippetClassificationLightningDataset
from gesture_classification.model import LitModel
from gesture_classification.helpers import (
    get_num_frames, get_accelerator, parse_use_keypoints
)

logger = logging.getLogger(__name__)

slurm_job_id = os.getenv("SLURM_JOB_ID")
start_time = strftime("%d_%m_%Y_%H:%M:%S", localtime())

def trainer(cfg):
    logger = logging.getLogger(__name__)
    logger.info(cfg)
    model_name = cfg.model.architecture

    use_audio = cfg.model.use_audio

    precision = "bf16-mixed" if cfg.common.fp16 else 32
    SEED = cfg.common.seed
    
    dataset_home = cfg.dataset.path
    subsample_rate = cfg.dataset.preprocessing.subsample_rate

    epochs = cfg.training.epochs
    batch_size = cfg.training.batch_size

    scheduler_name = cfg.scheduler.name
    scheduler_params = cfg.scheduler

    loss_function_name = cfg.criterion.name

    learning_rate = cfg.optimiser.learning_rate
    weight_decay = cfg.optimiser.weight_decay
    
    nodes = cfg.distributed_training.nodes
    gpus = cfg.distributed_training.gpus
    num_workers = cfg.distributed_training.num_workers * gpus
    accumulate_batches = cfg.distributed_training.accumulate_batches

    use_keypoints = parse_use_keypoints(cfg.features.use_keypoints)

    save_top_k = cfg.checkpoints.save_top_k
    
    logger_name = cfg.logging.log_name
    logger_folder = cfg.logging.full_path

    resize_to = cfg.augmentations.resize_to

    focal_gamma = 1

    seed_everything(SEED, workers=True)
    
    num_frames = get_num_frames(dataset_home, subsample_rate)
    logger.info(num_frames)
    accelerator = get_accelerator()
    dm = SnippetClassificationLightningDataset(
        dataset_home,
        batch_size,
        num_workers,
        subsample_rate,
        num_frames,
        resize_to,
        use_audio, 
        use_keypoints,
        )
    model = LitModel(
        model_name,
        num_frames,
        learning_rate,
        weight_decay,
        loss_function_name,
        focal_gamma,
        scheduler_name,
        scheduler_params,
        use_audio,
        use_keypoints
        )
    model.save_hyperparameters(cfg)
    checkpoint_f1 = ModelCheckpoint(
        save_top_k=save_top_k, mode="max", monitor="val_f1",
        filename="checkpoint-{epoch:02d}-{val_f1:.2f}"
        )
    checkpoint_acc = ModelCheckpoint(
        save_top_k=save_top_k, mode="max", monitor="val_acc",
        filename="checkpoint-{epoch:02d}-{val_acc:.2f}"
        )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger(
        name=f"id_{slurm_job_id}_{start_time}_{logger_name}",
        save_dir=logger_folder)
    trainer = Trainer(
        accelerator=accelerator,
        devices=gpus,
        num_nodes=nodes,
        max_epochs=epochs,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=precision,
        enable_progress_bar=False,
        callbacks=[checkpoint_f1, checkpoint_acc, lr_monitor],
        logger=logger,
        accumulate_grad_batches=accumulate_batches,
        num_sanity_val_steps=0,
        )
    trainer.fit(model, datamodule=dm)

