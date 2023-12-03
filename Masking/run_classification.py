from argparse import ArgumentParser
import logging
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from gesture_classification.datasets import SnippetClassificationLightningDataset
from gesture_classification.model import LitModel
from gesture_classification.helpers import (
    get_num_frames, get_subsample_rate,
    get_accelerator, parse_use_keypoints
)
from gesture_classification.constants import SEED


logging.getLogger("lightning").setLevel(logging.WARNING)


def main(args):
    dataset_home = args.dataset_home
    logger_name = args.logger_name
    logger_folder = args.logger_folder
    batch_size = args.batch_size
    model_name = args.model_name
    pretrained_model = args.pretrained_model
    nodes = args.nodes
    gpus = args.gpus
    epochs = args.epochs
    num_workers = args.workers_per_gpu * gpus
    accumulate_batches = args.accumulate_batches
    learning_rate = args.learning_rate
    save_top_k = args.save_top_k
    precision = args.precision
    #zero_normalisation = args.zero_normalisation
    use_keypoints = parse_use_keypoints(args.use_keypoints)
    seed_everything(SEED, workers=True)
    subsample_rate = get_subsample_rate(dataset_home)
    num_frames = get_num_frames(dataset_home, subsample_rate)
    accelerator = get_accelerator()
    dm = SnippetClassificationLightningDataset(
        dataset_home, 
        batch_size, 
        num_workers, 
        subsample_rate, 
        num_frames,
        #zero_normalisation,
        use_keypoints, 
        )
    model = LitModel(
        model_name, pretrained_model, num_frames, learning_rate,epochs, use_keypoints,
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
    

    """writer = SummaryWriter(logger_folder)
    examples=iter(dm.test_dataloader())
    print(type(examples))
    example_data,example_label =examples.next()
    print(example_data.shape,example_label.shape)
    frame_index=0
    frame=example_data[frame_index]
    img_grid = torchvision.utils.make_grid(frame.permute(0,3,1,2))
    writer.add_image('image_grid',img_grid)"""

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
        log_every_n_steps=50,
        accumulate_grad_batches=accumulate_batches,
        )
    trainer.fit(model, datamodule=dm)

parser = ArgumentParser()
parser.add_argument("--dataset-home", type=str)
parser.add_argument("--logger-name", type=str, default="gesture_classification")
parser.add_argument("--logger-folder", type=str)
parser.add_argument("--pretrained-model", type=str, default="")
parser.add_argument("--pretrained-dataset", type=str, default="ssv2")
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--precision", type=int, default=16)
parser.add_argument("--save-top-k", type=int, default=1)
parser.add_argument("--learning-rate", type=float, default=1e-5)
parser.add_argument("--nodes", type=int, default=1)
parser.add_argument("--gpus", type=int, default=8)
parser.add_argument("--workers-per-gpu", type=int, default=16)
parser.add_argument("--accumulate-batches", type=int, default=8)
parser.add_argument("--epochs", type=int)
#parser.add_argument("--zero_normalisation", type=bool,default=True)
parser.add_argument("--use-keypoints", default=0)
parser.add_argument("--model-name", type=str)
args = parser.parse_args()

if __name__ == "__main__":
    main(args)