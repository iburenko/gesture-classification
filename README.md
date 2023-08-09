# gesture_classification
-----
## Contents:
* [Dataset](#dataset)
    * [Description](#description)
    * [Folder structure](#folder-strucure)
* [Running a script](#running-a-training-script)
* [Training details](#training-details)


## Dataset

### Description

Home folder of a dataset has the following pattern:

```ellen_show_length_%d_sample_rate_%d_iou_%f```

- **Length** is a duration of snippets in ms;
- **Sample rate** is an integer. In practice it is 1, 2 or 4;
- **IoU** is a float. At least ```iou``` \% of a snippet should 
    have an intersection with ground truth to be considered as a positive example.

**NotaBene**: [Some](helpers.py#L8) [parts](helpers.py#L28) 
    of the code use the name format of the home folder.

### Folder strucure

Folder containing a dataset has the following structure:

```
ellen_show_length_%d_sample_rate_%d_iou_%f:
+-- train
    +-- gesture
        +-- train_snippet_id1.npz
        +-- train_snippet_id2.npz
        +-- ...
    +-- nongesture
        +-- train_snippet_id1.npz
        +-- train_snippet_id2.npz
        +-- ...
+-- val
+-- test
```
```Val``` and ```test``` folders have the same structure as ```train``` do.

Each ```*.npz``` file contains a dictionary with two keys:

- "video": (np.ndarray) A numpy array of shape (39, 320, 320, 3) obtained from a snippet;

- "keypoints: (np.ndarray) A numpy array of shape (39,320,320) obtained from an openpose output.
    Contains two-dimensional coordinates of joints.

## Running a training script

Command line arguments are understandable, but it is important to keep in mind that the script was run with SLURM, without an access to standalone videocards.

Required parameters:

- ```dataset-home```: path to a data folder;
- ```logger-folder```: path to a log folder;
- ```epochs```: how long to train;
- ```model-name```: which model to finetune.

**Notes**:

- Both [```timesformer```](https://github.com/facebookresearch/TimeSformer) and 
        [```videomae```](https://huggingface.co/docs/transformers/model_doc/videomae)
        works. But ```videomae``` is more stable during training and shows better results.
- It is crucial to use **pretrained** networks. Now [```ssv2```](https://developer.qualcomm.com/software/ai-datasets/something-something) 
    is used as a pretraining dataset, but other options are eligible.

Run training with the following command:

```python run_classification.py \
    --dataset-home=$DATASET_HOME \
    --logger-folder=$LOGGER_FOLDER \
    --epochs=$EPOCHS \
    --model-name=$MODEL_NAME
```

## Training details

There is a learning rate scheduler that divides the initial learning rate by 5 at the 35th and 80th epochs. It is not configurable in the current version of the code.