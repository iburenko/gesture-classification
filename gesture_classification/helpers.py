import os
import sys
import math

import torch

from .datasets import (
    SnippetClassificationLightningDataset,
    FrameBasedClassificationLightningDataset
)
from .constants import FPS

def LINE():
    return sys._getframe(1).f_lineno

def get_num_frames(dataset_path:str, subsample_rate:int) -> int:
    """
    Calculate number of frames.

    Args:
        dataset_path ('str' or 'PoxisPath', *required*):
            Path to a dataset.
        subsample_rate ('int', *required*):
            Take every ```subsample_rate``` frames to a batch.
    """
    
    if dataset_path.endswith("/"):
        basename = basename.rstrip("/")
    basename = os.path.basename(dataset_path)
    path_splitted = basename.split("_")
    snippet_length = float(path_splitted[3])
    num_frames = math.ceil(FPS * snippet_length / 1000)
    num_frames //= subsample_rate
    return num_frames

def get_subsample_rate(dataset_path:str) -> int:
    """
    Calculate subsample rate.

    Args:
        dataset_path ('str' or 'PoxisPath', *required*):
            Path to a dataset.
    """

    basename = os.path.basename(dataset_path)
    path_splitted = basename.split("_")
    return int(path_splitted[6])

def get_accelerator() -> str:
    """Get accelerator"""

    return "gpu" if torch.cuda.is_available() else "cpu"

def parse_use_keypoints(use_keypoints_input):
    """
    Decide whether to use openpose keypoints or not.

    Args:
        use_keypoints_input ('int' or 'str', *required*):
            Whether to use openpose keypoints.
    """

    use_keypoints = False
    if isinstance(use_keypoints_input, str):
        if use_keypoints_input.lower() in ["1", "true"]:
            use_keypoints = True
        elif use_keypoints_input.lower() == "only":
            use_keypoints = "only"
    elif isinstance(use_keypoints_input, int):
        if use_keypoints_input == 1:
            use_keypoints = True
    return use_keypoints

def get_dm(
        frame_based,
        data_home,
        target_size,
        use_audio,
        batch_size,
        num_workers,
        *args,
        **kwargs
        ):
    if frame_based:
        split_path = kwargs["cfg"].dataset.split_path
        csv_home = kwargs["cfg"].dataset.csv_home
        snippet_len = kwargs["cfg"].dataset.snippet_len
        dm = FrameBasedClassificationLightningDataset(
            data_home,
            split_path,
            csv_home,
            use_audio=use_audio,
            snippet_len=snippet_len,
            target_size=target_size,
            batch_size=batch_size,
            num_workers=num_workers,
            train_steps=kwargs["cfg"].training.train_steps,
            val_steps=kwargs["cfg"].training.val_steps,
        )
    else:
        subsample_rate = kwargs["cfg"].dataset.preprocessing.subsample_rate
        num_frames = get_num_frames(data_home, subsample_rate)
        use_keypoints = parse_use_keypoints(kwargs["cfg"].features.use_keypoints)
        dm = SnippetClassificationLightningDataset(
            data_home,
            batch_size,
            num_workers,
            subsample_rate,
            num_frames,
            target_size,
            use_audio, 
            use_keypoints,
        )
    return dm