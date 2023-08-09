import os
import math

import torch

from .constants import FPS

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