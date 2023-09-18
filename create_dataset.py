from os import listdir as ls, path, makedirs
import math
import json
import argparse
import subprocess
from natsort import natsorted
import cv2
from itertools import product
import pandas as pd
import numpy as np
from moviepy.editor import VideoFileClip

FPS = 29.97

# parser = argparse.ArgumentParser()
# parser.add_argument('--tmpdir-home', type=str)
# parser.add_argument('--split-home', type=str)
# parser.add_argument('--csv-home', type=str)
# parser.add_argument('--videos-home', type=str)
# parser.add_argument('--length-msec', type=int)
# parser.add_argument('--threshold', type=float)
# args = parser.parse_args()

ffmpeg = "/home/atuin/b105dc/data/software/ffmpeg/ffmpeg"

class args:
    tmpdir_home = "/scratch/b105dc10"
    split_home = "/home/atuin/b105dc/data/datasets/gestures/split.json"
    csv_home = "/home/titan/sles/sles000h/input_files_ellen_with_csv"
    videos_home = "/home/titan/sles/sles000h/input_files_ellen_with_csv"
    length_msec = 1300
    threshold = 0.55

with open(args.split_home, 'r') as file:
    split = json.load(file)
train = split['train']
val = split['val']
test = split['test']

all_csvs = natsorted([elem for elem in ls(args.csv_home) if elem.endswith('csv')])

node_path = args.tmpdir_home
keypoints_home = path.join(node_path, 'publish')
length_msec = args.length_msec
thr = args.threshold

home_path = node_path + "/" + \
            "ellen_show_length_" + str(length_msec) + \
            "_iou_thr_" + str(thr) + "_video_and_audio/"
for elem in product([home_path], ['train', 'val', 'test'], ['gesture', 'nongesture']):
    makedirs(path.join(*elem), exist_ok=True)

def sec2frame(sec):
    return math.ceil(sec * FPS)

def framelist2tensor(frame_list, video_shape, output_shape):
    tensor = np.zeros((len(frame_list), *output_shape[::-1]))
    rescale_x = output_shape[1]/video_shape[1]
    rescale_y = output_shape[0]/video_shape[0]
    limit_x = output_shape[1]
    limit_y = output_shape[0]
    for i, elem in enumerate(frame_list):
        data = json.load(open(elem))
        people = data['people']
        for person in people:
            keypoints = person['pose_keypoints_2d']
            y = np.array(keypoints[::3]) * rescale_y
            x = np.array(keypoints[1::3]) * rescale_x
            y = np.clip(y, 0, limit_y - 1)
            x = np.clip(x, 0, limit_x - 1)
            for x_coord, y_coord in zip(x,y):
                if x_coord == 0 and y_coord == 0:
                    continue
                tensor[i, round(x_coord), round(y_coord)] = 1
    return tensor

def get_label(csv_data, start_interval, end_interval, thr):
    length = 0
    for i in range(len(csv_data)):
        start_time = csv_data['Begin Time - msec'].iloc[i]
        end_time = csv_data['End Time - msec'].iloc[i]
        if start_interval <= start_time < end_interval <= end_time:
            length += (end_interval - start_time)
        if start_interval <= start_time < end_time <= end_interval:
            length += (end_time - start_time)
        if start_time <= start_interval <= end_time:
            length += (min(end_interval, end_time) - start_interval)
        if start_time > end_interval:
            break
    iou = length/(end_interval - start_interval)
    if iou > thr:
        return True
    else:
        return False

def subclip2tensor(subclip, num_frames):
    frame_tensor = np.zeros((num_frames, 320, 320, 3))
    frame_iterator = subclip.iter_frames()
    for i, frame in enumerate(frame_iterator):
        frame_tensor[i] = cv2.resize(frame, (320, 320))
    return frame_tensor

def get_frame_list(frame_list, start_frame, end_frame, num_frames):
    overhead = 0
    if math.ceil(end_frame - start_frame) != num_frames:
        overhead = (start_frame + num_frames - end_frame)
        end_frame += overhead
    return frame_list[start_frame:end_frame]

def create_dataset(data, split):
    dataset_ind = 0
    clip_length = length_msec/1000
    num_frames = int(math.ceil(clip_length * 29.97))
    for elem in data:
        case_name = elem.split('.')[0]
        csv_file = [elem for elem in all_csvs if elem.startswith(case_name)][0]
        csv_data = pd.read_csv(path.join(args.csv_home,csv_file))
        csv_data = csv_data[csv_data['Duration - msec']>=150]
        csv_data = csv_data.reset_index()
        video_path = path.join(args.videos_home, elem)
        video_obj = VideoFileClip(video_path)
        video_duration = np.floor(video_obj.duration).astype('int')
        i = 0
        start_msec = 0
        all_files_home = path.join(keypoints_home, case_name, case_name+'_json')
        all_files = natsorted(ls(all_files_home))
        while start_msec/1000 + 2*clip_length < video_duration:
            start_sec, end_sec = i * clip_length, (i + 1) * clip_length
            start_msec = start_sec * 1000
            end_msec = end_sec * 1000
            # start_frame = sec2frame(i*clip_length)
            # end_frame = sec2frame((i+1)*clip_length)
            # frame_list = get_frame_list(all_files, start_frame, end_frame, num_frames)
            # frame_list = [path.join(all_files_home, elem) for elem in frame_list]
            # keypoints_tensor = framelist2tensor(frame_list, video_obj.size, (320,320))
            label = get_label(csv_data, start_msec, end_msec, thr)
            subclip = video_obj.subclip(i*clip_length, (i+1)*clip_length)
            # clip_tensor = subclip2tensor(subclip, num_frames)
            # input_item = {
            #     'video': clip_tensor.astype('uint8'),
            #     'keypoints': keypoints_tensor.astype('uint8'),
            # }
            elem_splitted = elem.split('_')
            suffix = elem_splitted[0] + '_' + elem_splitted[-1].split('.')[0]
            filename = str(dataset_ind).zfill(9) + '_' + suffix
            if label:
                filename = home_path + split + '/gesture/' + filename
            else:
                filename = home_path + split + '/nongesture/' + filename
            # np.savez_compressed(filename, **input_item)
            subclip.write_videofile(filename + ".mp4", audio=False)
            subclip.audio.write_audiofile(filename + ".wav", fps=16000)
            dataset_ind += 1
            i += 1
            if dataset_ind == 64:
                break


if __name__ == "__main__":
    print("Start creating train dataset!")
    create_dataset(split["train"], "train")
    print("Start creating val dataset!")
    create_dataset(split["val"], "val")
    # print("Start creating test dataset!")
    # create_dataset(split["test"], "test")
