import logging
import sys

# from timesformer.models.vit import TimeSformer
import pytorch_lightning as pl
import torch
from torch import nn
from torchvision import models
from transformers import VideoMAEConfig, VideoMAEForVideoClassification
from transformers import AutoProcessor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model
from einops import rearrange
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score,
    BinaryJaccardIndex, BinaryPrecision, BinaryRecall
)
from .models.resnet import ResEncoder
from .helpers import LINE

logger = logging.getLogger(__name__)

from .loss_helpers import LossFunction


class LitModel(pl.LightningModule):
    """
    Lightning wrapper for training.

    Args:
        model_name ('str', *required*):
            What off-the-shelf model to use. Can be one of:
                - "timesformer".
                - "videomae".
        num_frames ('int', *required*):
            Number of frames to use in a batch for each video.
        learning_rate ('float', *required*):
            Magnitude of the step of gradinet descent.
        use_keypoints ('int' or 'str', *optional*, defaults to '0')
            Whether to use coordinates of openpose keypoints. Can be one of:
                - '0' or 'false' or 'False'.
                - '1' or 'true' or 'True.
                - 'only'.
    """

    def __init__(
        self, 
        model_name, 
        num_frames, 
        learning_rate, 
        weight_decay, 
        loss_function_name, 
        focal_gamma,
        scheduler_name, 
        scheduler_params,
        use_audio,
        use_keypoints,
        ):
        super().__init__()
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model = self.configure_model(
            model_name, use_keypoints, num_frames
            )
        self.use_audio = use_audio
        if self.use_audio:
            self.audio_processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.normalize = self._normalize
        self.reshape = self._reshape
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params
        self.criterion = LossFunction(
            (loss_function_name, focal_gamma)).loss_function
        self.train_precision = BinaryPrecision()
        self.train_accuracy = BinaryAccuracy()
        self.train_recall = BinaryRecall()
        self.train_f1 = BinaryF1Score()
        self.train_iou = BinaryJaccardIndex()
        self.val_precision = BinaryPrecision()
        self.val_accuracy = BinaryAccuracy()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()
        self.val_iou = BinaryJaccardIndex()
        self.use_keypoints = use_keypoints
        if self.use_audio:
            self.linear = nn.Linear(2048+768, 1)

    def _normalize(self, x):
        if self.use_keypoints in [0, 1]:
            x[:3] = x[:3] - 0.5
        return x

    def _reshape(self, x):
        if self.model_name in ["timesformer", "resnet50_3d"]:
            x = rearrange(x, "b t h w c -> b c t h w")
        elif self.model_name == "videomae":
            x = rearrange(x, "b t h w c -> b t c h w")
        return x

    def forward(self, video_data, audio_data):
        logits = self.model(video_data)
        if self.model_name == "videomae":
            logits = logits.logits
        if self.use_audio:
            audio_data = [elem.tolist() for elem in audio_data]
            inputs = self.audio_processor(audio_data, sampling_rate=16000, return_tensors="pt")
            with torch.no_grad():
                outputs = self.audio_model(inputs["input_values"].half().to('cuda'))
            last_hidden_states = outputs.last_hidden_state
            last_hidden_states = last_hidden_states.mean(axis=1)
            last_hidden_states = torch.cat([logits, last_hidden_states], axis=-1)
            last_hidden_states = self.linear(last_hidden_states)
            return last_hidden_states
        return logits

    def training_step(self, batch, batch_idx) -> float:
        video_data, audio_data, y = batch
        video_data = self.reshape(video_data)
        y_hat = self(video_data, audio_data)[:,0]
        loss = self.criterion(y_hat, y.float())
        probs = torch.sigmoid(y_hat)
        train_acc = self.train_accuracy(probs, y)
        train_prec = self.train_precision(probs, y)
        train_rec = self.train_recall(probs, y)
        train_f1 = self.train_f1(probs, y)
        train_iou = self.train_iou(probs, y)
        self.log("train_loss", loss, on_epoch=True, sync_dist=True)
        # self.log("train_acc_step", train_acc)
        # self.log("train_prec_step", train_prec)
        # self.log("train_rec_step", train_rec)
        # self.log("train_f1_step", train_f1)
        # self.log("train_iou_step", train_iou)
        return loss

    def on_train_epoch_end(self, outputs=None) -> None:
        self.log("train_acc", self.train_accuracy.compute(), sync_dist=True)
        self.log("train_prec", self.train_precision.compute(), sync_dist=True)
        self.log("train_rec", self.train_recall.compute(), sync_dist=True)
        self.log("train_f1", self.train_f1.compute(), sync_dist=True)
        self.log("train_iou", self.train_iou.compute(), sync_dist=True)
        self.train_accuracy.reset()
        self.train_precision.reset()
        self.train_recall.reset()
        self.train_f1.reset()
        self.train_iou.reset()

    def validation_step(self, batch, batch_idx) -> dict:
        video_data, audio_data, y = batch
        video_data = self.reshape(video_data)
        y_hat = self(video_data, audio_data)[:,0]
        loss = self.criterion(y_hat, y.float())
        self.log("val_loss", loss, sync_dist=True)
        probs = torch.sigmoid(y_hat)
        val_acc = self.val_accuracy.update(probs, y)
        val_prec = self.val_precision.update(probs, y)
        val_rec = self.val_recall.update(probs, y)
        val_iou = self.val_iou.update(probs, y)
        val_f1 = self.val_f1.update(probs, y)
        return {
            "loss": loss,
            "acc": val_acc,
            "rec": val_rec,
            "prec": val_prec,
            "f1": val_f1,
            "iou": val_iou
        }

    def on_validation_epoch_end(self, outputs=None) -> None:
        self.log("val_acc", self.val_accuracy.compute(), sync_dist=True)
        self.log("val_prec", self.val_precision.compute(), sync_dist=True)
        self.log("val_rec", self.val_recall.compute(), sync_dist=True)
        self.log("val_f1", self.val_f1.compute(), sync_dist=True)
        self.log("val_iou", self.val_iou.compute(), sync_dist=True)
        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_iou.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
            )
        scheduler = self.configure_scheduler(
            optimizer, 
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }
    
    def configure_scheduler(
            self, 
            optimizer,
            ):
        if self.scheduler_name == "multi-step-lr":
            params = {
                "optimizer": optimizer,
                "milestones": self.scheduler_params.steps,
                "gamma": self.scheduler_params.gamma
            }
            scheduler = torch.optim.lr_scheduler.MultiStepLR(**params)
        elif self.scheduler_name == "warmup":
            linear = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=20
            )
            exponential = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.96
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[linear, exponential],
                milestones=[20]
            )
        return scheduler

    def configure_model(
            self, model_name, use_keypoints, num_frames):
        if use_keypoints == False:
            in_chans = 3
        elif use_keypoints == True:
            in_chans = 4
        elif use_keypoints == "only":
            in_chans = 1
        if model_name == "timesformer":
            return NotImplemented
            model = TimeSformer(
                img_size=224,
                num_classes=1,
                num_frames=num_frames,
                attention_type="divided_space_time",
                pretrained_model=pretrained_model,
                in_chans=in_chans
            )
        elif model_name == "r2plus1":
            raise NotImplemented
            model = models.video.r2plus1d_18()
            model.fc = torch.nn.Linear(512, 1)
        elif model_name == "videomae":
            config = VideoMAEConfig(
                num_frames=num_frames,
                qkv_bias=False,
                num_labels=1,
            )
            model = VideoMAEForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-ssv2",
                config=config,
                ignore_mismatched_sizes=True
                )
        elif model_name == "resnet50_3d":
            pretrained_weights = torch.load("/home/hpc/b105dc/b105dc10/.cache/torch/hub/checkpoints/resnet50_a1_0-14fe96d1.pth")
            model = ResEncoder("prelu", None)
            model.trunk.load_state_dict(pretrained_weights, strict=False)
            logger.info(model)
        return model