from os import path, listdir as ls
import random
from pathlib import PosixPath
import logging

from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from vidaug import augmentors as va
import torch
import cv2
from moviepy.editor import VideoFileClip
from scipy.io import wavfile

logger = logging.getLogger(__name__)


class SnippetClassificationLightningDataset(LightningDataModule):
    """
    Lighning wrapper for a dataset.

    Args:
        home ('str' or 'PosixPath', *required*):
            Path to the folder where a dataset is stored.
        batch_size ('int', *required*):
            Batch size.
        num_workers ('int', *required*):
            Number of loader worker processes.
        subsample_rate ('int', *required*):
            Take every ```subsample_rate``` frame to a batch.
        num_frames ('int', *required*):
            Number of frames from a video to use in a batch.
        use_keypoints ('int', *optional*, defaults to '0'):
            Whether to use coordinates of openpose keypoints. Can be one of:
                - '0' or 'false' or 'False'.
                - '1' or 'true' or 'True.
                - 'only'.
    """

    def __init__(
        self, 
        home: str, 
        batch_size: int, 
        num_workers: int,
        subsample_rate: int,
        num_frames: int,
        resize_to: int, 
        use_audio: int,
        use_keypoints: int=0,
        ):
        assert isinstance(home, str) or isinstance(home, PosixPath)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(num_workers, int) and num_workers >= 0
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self.home = home
        logger.info(self.home)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_audio = use_audio
        self.use_keypoints = use_keypoints
        self.subsample_rate = subsample_rate
        self.resize_to = resize_to
        self.num_frames = num_frames
        self.train_path = path.join(self.home, "train")
        self.train_data = SnippetClassificationDataset(
            self.train_path, 
            "train", 
            self.subsample_rate, 
            self.num_frames,
            self.resize_to,
            self.use_audio,
            self.use_keypoints
            )
        self.val_path = path.join(self.home, "val")
        self.val_data = SnippetClassificationDataset(
            self.val_path, 
            "val", 
            self.subsample_rate, 
            self.num_frames,
            self.resize_to,
            self.use_audio,
            self.use_keypoints
            )
        # self.test_path = path.join(self.home, "test")
        # self.test_data = SnippetClassificationDataset(
        #     self.test_path, 
        #     "test", 
        #     self.subsample_rate,
        #     self.num_frames,
        #     self.resize_to,
        #     self.use_audio,
        #     self.use_keypoints
        #     )
        self.prepare_data_per_node = False
        self._log_hyperparams = False

    def train_dataloader(self):
        weights = self._get_weights()
        sampler = WeightedRandomSampler(
            weights,
            len(weights),
            replacement=True,
        )
        train_loader = DataLoader(
            self.train_data, 
            sampler=sampler,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # shuffle=True,
            pin_memory=True
            )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_data,
            batch_size=4*self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_data,
            batch_size=4*self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return test_loader

    def _get_weights(self):
        weights = [1] * len(self.train_data.data_list)        
        nongesture_len = len(
            [
                elem for elem in self.train_data.data_list 
                if "nongesture" in elem
            ])
        gesture_len = len(self.train_data) - nongesture_len
        gesture_weight = 1. / 2 / gesture_len
        nongesture_weight = 1. / 2 / nongesture_len
        for i, elem in enumerate(self.train_data.data_list):
            if "nongesture" in elem:
                weights[i] = nongesture_weight
            else:
                weights[i] = gesture_weight
        return weights        


class SnippetClassificationDataset(Dataset):
    """
    Dataset class

    Args:
        home ('str' or 'PosixPath', *required*):
            Path to the folder where a dataset is stored.
        split ('str', *required*):
            Whether to use train, validation or test data split. Can be one of:
                - 'test.
                - 'val'.
                - 'test'.
        subsample_rate ('int', *required*):
            Take every ```subsample_rate``` frame to a batch.
        num_frames ('int', *required*):
            Number of frames from a video to use in a batch.
        use_keypoints ('int', *optional*, defaults to '0'):
            Whether to use coordinates of openpose keypoints. Can be one of:
                - '0' or 'false' or 'False'.
                - '1' or 'true' or 'True.
                - 'only'.
        
    """

    def __init__(
        self, 
        home,
        split: list, 
        subsample_rate: int, 
        num_frames: int,
        resize_to: int, 
        use_audio: int, 
        use_keypoints: int=0, 
        ):
        assert isinstance(home, str) or isinstance(home, PosixPath)
        assert split in ["train", "val", "test"]
        super().__init__()
        self.home = home
        self.split = split
        self.resize_to = resize_to
        self.use_keypoints = use_keypoints
        self.subsample_rate = subsample_rate
        self.num_frames = num_frames
        self.use_audio = use_audio
        gesture_path = path.join(self.home, "gesture")
        nongesture_path = path.join(self.home, "nongesture")
        gesture_list = [
            "gesture/" + elem for elem in ls(gesture_path)
        ]
        nongesture_list = [
            "nongesture/" + elem for elem in ls(nongesture_path)
        ]
        gesture_list = list({elem[:-3] for elem in gesture_list})
        nongesture_list = list({elem[:-3] for elem in nongesture_list})
        self.gesture_len = len(gesture_list)
        all_gestures = gesture_list + nongesture_list
        self.data_list = sorted(all_gestures, key=lambda x:x.split("/")[1])
        self.transform = Augmentation(resize_to, use_keypoints)
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        full_item_path = path.join(self.home, self.data_list[idx])
        label = 0 if "nongesture" in self.data_list[idx] else 1
        if self.use_audio:
            data = self._load_audio_video_item(full_item_path)
        else:
            data = self._load_npz_item(full_item_path)
        audio_data = data["audio"] if self.use_audio else np.zeros(1)
        if self.use_keypoints == 0:
            data = data["video"]/255
        elif self.use_keypoints == 1:
            video = data["video"]/255
            keypoints = np.expand_dims(data["keypoints"], -1)
            data = np.concatenate([video, keypoints], axis=-1)
        elif self.use_keypoints == "only":
            data = np.expand_dims(data["keypoints"], -1)
        if self.split == "train":
            data = np.stack(self.transform(data, "train"), axis=0)
        else:
            data = np.stack(self.transform(data, self.split), axis=0)
        if self.subsample_rate > 1:
            len_data = len(data) - self.num_frames % 2
        else:
            len_data = len(data)
        data = data.astype("float32")[:len_data:self.subsample_rate]
        return data, audio_data, label
    
    def _load_npz_item(self, full_item_path):
        data = np.load(full_item_path+"npz")
        data = dict(data)
        return data

    def _load_audio_video_item(self, full_item_path):
        audio_fp = full_item_path + "wav"
        video_fp = full_item_path + "mp4"
        sr, audio_data = wavfile.read(audio_fp)
        if len(audio_data.shape) == 2:
            audio_data = audio_data.mean(axis=1)
        if len(audio_data) == 20800:
            audio_data = audio_data[:-1]        
        video = VideoFileClip(video_fp)
        number_of_frames = int(np.ceil(video.fps * video.duration))
        video_data = np.zeros((number_of_frames, 320, 320, 3))
        for i, frame in enumerate(video.iter_frames()):
            frame = cv2.resize(frame, (320, 320)).astype('uint8')
            video_data[i] = frame
        return {
            "audio": audio_data,
            "video": video_data
        }


class Augmentation(LightningModule):
    """
    Controls data augmentation.

    Args:
        use_keypoints ('int', *optional*, defaults to '0'):
            Whether to use openpose keypoints. Can be one of:
                - '0' or 'false' or 'False'.
                - '1' or 'true' or 'True.
                - 'only'.
    """

    def __init__(self, resize_to: int, use_keypoints: bool=0):
        super().__init__()
        self.resize_to = resize_to
        self.use_keypoints = use_keypoints
        self.train_transforms = va.Sequential(
            [
                va.RandomCrop(size=(300, 300)),
                # va.RandomResize(rate=.75),
                self.resize_video,
                self.coin_toss(va.RandomRotate(degrees=10)),
                self.coin_toss(va.HorizontalFlip())
            ]
        )
        self.val_transforms = va.Sequential(
            [
                # va.CenterCrop(300),
                self.resize_video
            ]
        )
        
    @torch.no_grad()
    def forward(self, x, split):
        if split == "train":
            return self.train_transforms(x)
        elif split in ["val", "test"]:
            return self.val_transforms(x)

    def coin_toss(self, aug):
        return va.Sometimes(0.5, aug)

    def channel_dropout(self, x):
        if self.use_keypoints == "only":
            return x
        if random.random() >= 0.5 and self.use_keypoints >= 0:
            channel = random.randint(0,2)
            value = random.random()
            x[..., channel] = value
        return x

    def resize_video(self, video, size=224):
        return [cv2.resize(frame, (size, size)) for frame in video]