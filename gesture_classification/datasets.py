from os import path, listdir as ls, getenv
import subprocess
from dataclasses import dataclass
import math
import random
from pathlib import PosixPath
import logging
import json

from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from vidaug import augmentors as va
import torch
from torchvision.io import read_video
from torchvision import transforms as ImT
import torchaudio.transforms as AT
from pytorchvideo import transforms as VT
import cv2
from moviepy.editor import VideoFileClip
from scipy.io import wavfile
import pandas as pd

logger = logging.getLogger(__name__)
ffmpeg = "/home/atuin/b105dc/data/software/ffmpeg/ffmpeg"
ffprobe = "/home/atuin/b105dc/data/software/ffmpeg/ffprobe"


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
        target_size: int, 
        use_audio: int,
        use_keypoints: int=0,
        *args,
        **kwargs
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
        self.target_size = target_size
        self.num_frames = num_frames
        self.train_path = path.join(self.home, "train")
        self.train_data = SnippetClassificationDataset(
            self.train_path, 
            "train", 
            self.subsample_rate, 
            self.num_frames,
            self.target_size,
            self.use_audio,
            self.use_keypoints
            )
        self.val_path = path.join(self.home, "val")
        self.val_data = SnippetClassificationDataset(
            self.val_path, 
            "val", 
            self.subsample_rate, 
            self.num_frames,
            self.target_size,
            self.use_audio,
            self.use_keypoints
            )
        # self.test_path = path.join(self.home, "test")
        # self.test_data = SnippetClassificationDataset(
        #     self.test_path, 
        #     "test", 
        #     self.subsample_rate,
        #     self.num_frames,
        #     self.target_size,
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
        target_size: int, 
        use_audio: int, 
        use_keypoints: int=0, 
        ):
        assert isinstance(home, str) or isinstance(home, PosixPath)
        assert split in ["train", "val", "test"]
        super().__init__()
        self.home = home
        self.split = split
        self.target_size = target_size
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
        self.transform = Augmentation(target_size, use_keypoints)
        
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

    def __init__(self, target_size: int, use_keypoints: bool=0):
        super().__init__()
        self.target_size = target_size
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
    

@dataclass
class FrameBasedDataclass:
    data_home: str
    split_path: str
    csv_home: str
    split: str
    use_audio: bool = True
    snippet_len: int = 10
    target_size: int = 112

class FrameBasedClassificationLightningDataset(LightningDataModule):
    def __init__(
            self,
            data_home,
            split_path,
            csv_home,
            use_audio=True,
            snippet_len=10,
            target_size=112,
            batch_size=2,
            num_workers=16,
            train_steps=1024,
            val_steps=256,
            *args,
            **kwargs
            ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        logger.info("Starting loading validation dataset!")
        self.val_ds = FrameBasedClassificationDataset(
            data_home,
            split_path,
            csv_home,
            "val",
            use_audio=use_audio,
            snippet_len=snippet_len,
            target_size=target_size,
            train_steps=train_steps,
            val_steps=val_steps
        )
        logger.info("Finished. Validation dataset is loaded!")
        logger.info("Starting loading train dataset!")
        self.train_ds = FrameBasedClassificationDataset(
            data_home,
            split_path,
            csv_home,
            "train",
            use_audio=use_audio,
            snippet_len=snippet_len,
            target_size=target_size,
            train_steps=train_steps,
            val_steps=val_steps
        )
        logger.info("Finished. Train dataset is loaded!")
        self.prepare_data_per_node = False
        self.allow_zero_length_dataloader_with_multiple_devices = True
        self._log_hyperparams = False

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader

class FrameBasedClassificationDataset(FrameBasedDataclass, Dataset):
    def __init__(
        self, 
        data_home, 
        split_path, 
        csv_home,
        split, 
        train_steps=1024,
        val_steps=256,
        use_audio=True,
        snippet_len=10,
        target_size=112
    ):
        super(Dataset).__init__()
        FrameBasedDataclass.__init__(
            self,
            data_home,
            split_path,
            csv_home,
            split,
            use_audio,
            snippet_len,
            target_size,
        )
        with open(self.split_path, "r") as file:
            split_files = json.load(file)
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.split_files = split_files[self.split]
        self.lengths = [self._get_video_length(elem) for elem in self.split_files]
        self.fps = self._get_fps(self.split_files[0])
        self._gts = list()
        self.resampler = AT.Resample(
            44100, 16000, 
            dtype=torch.float32)
        for video_name in self.split_files:
            csv_path = video_name[:-3] + "csv"
            gt = self._get_gt_from_csv(csv_path)
            self._gts.append(gt)
        # if self.split == "val":
        #     self.videos = [torch.randn(30000,640,640,3)*len(self.split_files)]
        #     self.audios = list()
        #     resizer = ImT.Resize(size=target_size, antialias=True)
        #     for i, video_name in enumerate(self.split_files):
        #         full_video_path = path.join(self.data_home, video_name)
        #         video_audio = read_video(full_video_path)
        #         video_data = video_audio[0]
        #         video_data = torch.einsum('ijkl -> iljk', video_data)
        #         if video_data.shape[0] > 15000:
        #             video_chunk1 = resizer(video_data[:15000])
        #             video_chunk2 = resizer(video_data[15000:])
        #             video_data = torch.cat([video_chunk1, video_chunk2], axis=0)
        #         else:
        #             video_data = resizer(video_data)
        #         video_data = torch.einsum('iljk -> ijkl', video_data)
        #         # self.videos.append(video_data)
        #         self.videos[i] = video_data[0]
        #         audio_data = self._get_audio_array(video_audio[1])
        #         self.audios.append(audio_data)
        #         csv_path = video_name[:-3] + "csv"
        #         gt = self._get_gt_from_csv(csv_path)
        #         self._gts.append(gt)
        #     self.cumsums = np.cumsum(
        #         [
        #             elem.shape[0] - math.ceil(self.snippet_len * self.fps) - 1 
        #             for elem in self.videos
        #         ])
        # self.sampler = self.set_sampler()
        self.sampler = self._random_sampler
        self.augmentation = self._get_augmentations()
        self.tmp = getenv("TMPDIR")
        print(self.tmp)
        
    def __len__(self):
        if self.split == "train":
            return self.train_steps
        else:
            return 128
            return self.cumsums[-1]
    
    def __getitem__(self, idx):
        video_data, audio_data, gt = self.sampler(idx)
        video_data = torch.einsum('thwc->cthw', video_data)
        video_data = self.augmentation(video_data)
        video_data = torch.einsum('cthw->thwc', video_data)
        return video_data, audio_data, gt
    
    def set_sampler(self):
        if self.split == "train":
            return self._random_sampler
        else:
            return self._consequential_sampler

    def _get_gt_from_csv(self, fn):
        csv_data = pd.read_csv(path.join(self.csv_home, fn))
        gt = list()
        for i in range(len(csv_data)):
            start_ms = csv_data['Begin Time - msec'].iloc[i]
            end_ms = csv_data['End Time - msec'].iloc[i]
            gt.append((start_ms, end_ms))
        return gt

    def _get_audio_array(self, audio_data):
        if self.use_audio:
            audio_data = self.resampler(audio_data)
            if len(audio_data.shape) == 2:
                audio_data = audio_data.mean(axis=0)
        else:
            audio_data = torch.tensor([0])
        return audio_data
    
    def _sec2tick(self, sec):
        tick = math.floor(16e3 * sec)
        return tick
    
    def _frame2tick(self, frame):
        sec = frame / self.fps
        tick = self._sec2ticks(sec)
        return tick    
    
    def _elemid2videoid(self, idx):
        for video_id, bound in enumerate(self.cumsums):
            if bound > idx:
                return video_id
        
    def _random_sampler(self, idx):
        ind = random.choice(range(len(self.split_files)))
        video_fp = path.join(self.data_home, self.split_files[ind])
        sample_fp = path.join(self.tmp, str(idx) + "_sample.mp4")
        video_len = self.lengths[ind]
        start_sec = random.uniform(0, video_len - 11)
        start_sec = round(start_sec, 2)
        cmd_ffmpeg = [
            ffmpeg, 
            "-i", video_fp, 
            "-ss", str(start_sec), 
            "-t", "10", 
            "-loglevel", "panic",
            sample_fp
        ]
        subprocess.call(cmd_ffmpeg)
        # start_frame = random.randint(
        #     0, 
        #     self.videos[video_id].shape[0] - math.ceil(self.snippet_len * self.fps) - 1
        # )
        # end_frame = start_frame + math.ceil(self.snippet_len * self.fps)
        media = read_video(sample_fp, pts_unit="sec")
        video_frames = media[0]
        audio_frames = self._get_audio_array(media[1])
        # video_frames = self.videos[video_id][start_frame:end_frame]
        # start_tick = self._sec2tick(start_sec)
        # end_tick = start_tick + self.snippet_len * 16000
        # audio_frames = self.audios[ind][start_tick:end_tick]
        # end_frame = math.ceil((start_sec + self.snippet_len) * self.fps)
        # print(ind, len(self.gts), len(self.split_files))
        # gt = self.gts(ind, start_sec + self.snippet_len)
        gt = self._get_gt(ind, start_sec + self.snippet_len)
        cmd_rm = ["rm", sample_fp]
        subprocess.call(cmd_rm)
        return video_frames, audio_frames, gt
    
    def _consequential_sampler(self, idx):
        video_id = self._elemid2videoid(idx)
        frame_id = self.cumsums[video_id-1] if video_id > 0 else 0
        # video = self.videos[video_id]
        start_frame = idx - frame_id
        start_sec = start_frame / self.fps
        end_frame = start_frame + math.ceil(self.snippet_len * self.fps)
        video_frames = self.videos[video_id][start_frame:end_frame]
        start_tick = self._frame2ticks(start_frame)
        end_tick = start_tick + self.snippet_len * 16000
        audio_frames = self.audios[video_id][start_tick:end_tick]
        # gt = self.gts[video_id][end_frame]
        gt = self._get_gt(video_id, start_sec + self.snippet_len)
        return video_frames, audio_frames, gt
    
    def _get_augmentations(self):
        mode = self.split
        num_samples = 300
        crop_size=self.target_size
        min_size=128
        max_size=256
        return VT.create_video_transform(
            mode, 
            num_samples=num_samples, 
            crop_size=crop_size, 
            min_size=min_size, 
            max_size=max_size
            )
    
    def _get_video_length(self, video_fp):
        video_fp = path.join(self.data_home, video_fp)
        length_cmd = [
            ffprobe, 
            "-v", "error", 
            "-show_entries", "format=duration", 
            "-of", "default=noprint_wrappers=1:nokey=1", video_fp
        ]
        output = subprocess.run(length_cmd, capture_output=True)
        length = float(output.stdout.decode().strip())
        logger.info(f"For file {video_fp} length = {length:.3f}")
        return length
    
    def _get_fps(self, video_fp):
        video_fp = path.join(self.data_home, video_fp)
        fps_cmd = [
            ffprobe, 
            "-v", "error", 
            "-select_streams", "v",
            "-show_entries", "stream=r_frame_rate", 
            "-of", "default=noprint_wrappers=1:nokey=1", video_fp
        ]
        output = subprocess.run(fps_cmd, capture_output=True)
        output = output.stdout.decode().strip()
        fps = round(float(eval(output)), 2)
        return fps
    
    def _get_gt(self, ind, end_sec):
        end_msec = end_sec * 1000
        for interval in self._gts[ind]:
            if interval[0] <= end_msec <= interval[1]:
                return True
        return False