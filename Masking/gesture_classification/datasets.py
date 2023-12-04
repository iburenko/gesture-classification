from os import path, listdir as ls
import random
from pathlib import PosixPath

from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from vidaug import augmentors as va
import torch
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from transformers import DetrImageProcessor
from transformers import DetrForObjectDetection


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
        #zero_normalisation: bool,
        use_keypoints: int=0,
        ):
        assert isinstance(home, str) or isinstance(home, PosixPath)
        assert isinstance(batch_size, int) and batch_size > 0
        assert isinstance(num_workers, int) and num_workers >= 0
        self.home = home
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_keypoints = use_keypoints
        self.subsample_rate = subsample_rate
        self.num_frames = num_frames
        #self.zero_normalisation =zero_normalisation
        self.train_path = path.join(self.home, "train")
        self.train_data = SnippetClassificationDataset(
            self.train_path, 
            "train", 
            self.subsample_rate, 
            self.num_frames,
            #self.zero_normalisation,
            self.use_keypoints
            )
        self.val_path = path.join(self.home, "val")
        self.val_data = SnippetClassificationDataset(
            self.val_path, 
            "val", 
            self.subsample_rate, 
            self.num_frames,
            #self.zero_normalisation,
            self.use_keypoints
            )
        self.test_path = path.join(self.home, "test")
        self.test_data = SnippetClassificationDataset(
            self.test_path, 
            "test", 
            self.subsample_rate,
            self.num_frames,
            #self.zero_normalisation,
            self.use_keypoints
            )
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
            batch_size=4 * self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_data,
            batch_size=4 * self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
        return test_loader

    def _get_weights(self):
        weights = [1] * len(self.train_data.data_list)        
        gesture_path = path.join(self.train_path, "gesture")
        nongesture_path = path.join(self.train_path, "nongesture")
        gesture_len = len(ls(gesture_path))
        nongesture_len = len(ls(nongesture_path))
        gesture_weight = 1. / 2 / gesture_len
        nongesture_weight = 1. / 2 / nongesture_len
        for i, elem in enumerate(self.train_data.data_list):
            if "nongesture" in elem:
                weights[i] = nongesture_weight
            else:
                weights[i] = gesture_weight
        #weights = 0.3
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
        #zero_normalisation: bool,
        use_keypoints: int=0, 
        ):
        assert isinstance(home, str) or isinstance(home, PosixPath)
        assert split in ["train", "val", "test"]
        super().__init__()
        self.home = home
        self.split = split
        self.use_keypoints = use_keypoints
        self.subsample_rate = subsample_rate
        self.num_frames = num_frames
        #self.zero_normalisation =zero_normalisation
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model= DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

        gesture_path = path.join(self.home, "gesture")
        nongesture_path = path.join(self.home, "nongesture")
        gesture_list = [
            "gesture/" + elem for elem in ls(gesture_path)
        ]
        nongesture_list = [
            "nongesture/" + elem for elem in ls(nongesture_path)
        ]
        self.gesture_len = len(gesture_list)
        all_gestures = gesture_list + nongesture_list
        self.data_list = sorted(all_gestures, key=lambda x:x.split("/")[1])
        self.transform = Augmentation(use_keypoints)
        
    def __len__(self):
        return len (self.data_list)
    

    def mask(self,data):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        stacked_results = []
        for img in data:
             #processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
            img_tensor = torch.tensor(img, device=device, dtype=torch.float32)

            encoding=self.processor(img_tensor, return_tensors="pt")
            #encoding.keys()
            #model=DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            with torch.no_grad():
                encoding = {k: v.to(device) for k, v in encoding.items()}
                outputs=self.model(**encoding)
            width, height, channels = img.shape
            postprocessed_outputs = self.processor.post_process_object_detection(outputs,
                                                                        target_sizes=[(height, width)],
                                                                        threshold=0.9)
            results = postprocessed_outputs[0]
            result=(results['labels']==1)
            scores_new=results['scores'][result]
            labels_new=results['labels'][result]
            boxes_new=results['boxes'][result]
            filter_data= {'scores':scores_new,
                    'labels' :labels_new,
                    'boxes':boxes_new}
            #plot_results(img, filter_data['scores'], filter_data['labels'], filter_data['boxes'])
            #image_array=np.array(img)
            mask=np.zeros_like(img)
            for box in boxes_new:
                xmin, ymin, xmax, ymax =box
                mask[int(ymin):int(ymax), int(xmin):int(xmax)]=1    
            result_array=img*mask
            stacked_results.append(result_array)
            #print(result_array)
            #result_image=Image.fromarray(result_array)
            #plt.imshow(result_array)
            #print('New_image_shape',result_array.shape)
            #plt.axis('off')  # Remove axis ticks and labels
            #plt.show()
        

        stacked_results=np.stack(stacked_results)
        #print("shape of stacked_results:",stacked_results.shape)

        return stacked_results




    def __getitem__(self, idx):
        #data = np.random.random([39,224,224,3]).astype(np.float32)
        #data = torch.tensor(data, dtype=torch.float32)
        #label = np.random.randint(0,2)

        full_item_path = path.join(self.home, self.data_list[idx])
        label = 0 if "nongesture" in self.data_list[idx] else 1
        data = np.load(full_item_path)
        data = dict(data)
        if self.use_keypoints == 0:
            
            data = data["masking"][:,:,:,:3]
            data = self.mask(data)
            data = data/255
            '''channel_1 =data["video_OF"][...,0:1]/255
            noise_channels =np.random.random([39,320,320,3]).astype(np.float32)
            data = np.concatenate([channel_1, noise_channels],axis=-1)'''
            #data = data["video_OF"][..., 3:6]/255
            '''if self.zero_normalisation:
                #vid_data = data["video_OF"][..., :3]
                OF_data = data["video_OF"][..., 3:6]
                #vid_mean=112.63 #mean of video considering train snippets
                #vid_std=63.44 #SD of video considering train snippets
                OF_mean=242.47 #mean of optical_flow considering train snippets
                OF_std=25.51   #SD of optical_flow considering train snippets
                #vid_norm=(vid_data-vid_mean)/vid_std
                OF_norm=(OF_data-OF_mean)/OF_std
                #data=np.concatenate([vid_norm,OF_norm],axis=-1)
                data=OF_norm'''
            '''else:
             data = data["video_OF"][..., 3:6]/255'''
            #print(data)
            #avail_channels = list(range(6))
            #sel_channels=random.sample(avail_channels,3)
            #data =data[:,:,:,sel_channels]


            #data = data[:,:,:,[0,1,-1]]
            #print(data)
            #print(data.shape)
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
        return data, label

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

    def __init__(self, use_keypoints=0):
        super().__init__()
        self.use_keypoints = use_keypoints
        self.train_transforms = va.Sequential(
            [
                self.channel_dropout,
                va.RandomResize(rate=0.1),
                va.RandomCrop(size=(224,224)),
                self.coin_toss(va.RandomRotate(degrees=5)),
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