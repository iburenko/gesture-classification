from argparse import ArgumentParser
import logging

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
from gesture_classification.constants import SEED

logging.getLogger("lightning").setLevel(logging.WARNING)

def main(args):
    dataset_home = args.dataset_home
    logger_name = args.logger_name
    logger_folder = args.logger_folder
    batch_size = args.batch_size
    model_name = args.model_name
    loss_function_name = args.loss_function_name
    pretrained_model = args.pretrained_model
    nodes = args.nodes
    gpus = args.gpus
    epochs = args.epochs
    num_workers = args.workers_per_gpu * gpus
    accumulate_batches = args.accumulate_batches
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    loss_function_name = args.loss_function_name
    focal_gamma = args.focal_gamma
    scheduler_name = args.scheduler_name
    scheduler_milestiones = args.scheduler_milestones
    scheduler_gamma = args.scheduler_gamma
    save_top_k = args.save_top_k
    precision = args.precision
    use_keypoints = parse_use_keypoints(args.use_keypoints)
    seed_everything(SEED, workers=True)
    subsample_rate = args.subsample_rate
    num_frames = get_num_frames(dataset_home, subsample_rate)
    accelerator = get_accelerator()
    dm = SnippetClassificationLightningDataset(
        dataset_home,
        batch_size,
        num_workers,
        subsample_rate,
        num_frames,
        use_keypoints,
        )
    model = LitModel(
        model_name,
        pretrained_model,
        num_frames,
        learning_rate,
        weight_decay,
        loss_function_name,
        focal_gamma,
        scheduler_name,
        scheduler_milestiones,
        scheduler_gamma,
        use_keypoints
        )
    model.save_hyperparameters(args)
    checkpoint_f1 = ModelCheckpoint(
        save_top_k=save_top_k, mode="max", monitor="val_f1",
        filename="checkpoint-{epoch:02d}-{val_f1:.2f}"
        )
    checkpoint_acc = ModelCheckpoint(
        save_top_k=save_top_k, mode="max", monitor="val_acc",
        filename="checkpoint-{epoch:02d}-{val_acc:.2f}"
        )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    logger = TensorBoardLogger(
        name=logger_name,
        save_dir=logger_folder)
    trainer = Trainer(
        accelerator=accelerator,
        devices=gpus,
        num_nodes=nodes,
        max_epochs=epochs,
        strategy=DDPStrategy(find_unused_parameters=False),
        precision=precision,
        enable_progress_bar=False,
        callbacks=[checkpoint_f1, checkpoint_acc, lr_monitor],
        logger=logger,
        accumulate_grad_batches=accumulate_batches,
        )
    trainer.fit(model, datamodule=dm)

parser = ArgumentParser()
parser.add_argument("--dataset-home", type=str)
parser.add_argument("--logger-name", type=str, default="gesture_classification")
parser.add_argument("--logger-folder", type=str)
parser.add_argument("--pretrained-model", type=str, default="")
parser.add_argument("--pretrained-dataset", type=str, default="ssv2")
parser.add_argument("--subsample-rate", type=int, default=1)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--precision", type=int, default=16)
parser.add_argument("--save-top-k", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=5e-5)
parser.add_argument("--weight-decay", type=float, default=1e-3)
parser.add_argument("--nodes", type=int, default=1)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--workers-per-gpu", type=int, default=16)
parser.add_argument("--accumulate-batches", type=int, default=8)
parser.add_argument("--epochs", type=int)
parser.add_argument("--use-keypoints", default=0)
parser.add_argument("--model-name", type=str, default="videomae")
parser.add_argument("--loss-function-name", type=str)
parser.add_argument("--focal-gamma", type=float, default=0.7)
parser.add_argument("--scheduler-name", type=str, default="multi-step-lr")
parser.add_argument("--scheduler-milestones", type=list, default=[35,80])
parser.add_argument("--scheduler-gamma", type=float, default=0.2)
args = parser.parse_args()
print(args)

if __name__ == "__main__":
    main(args)
