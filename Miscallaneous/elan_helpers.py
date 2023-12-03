from os import path

import numpy as np

FPS = 25
# FPS = 29.97

def filename2timestamps(filename):
    filename = path.basename(filename)
    filename_splitted = filename.split(")")
    if len(filename_splitted) > 1:
        filename = filename_splitted[1]
    else:
        filename = filename_splitted[0]
    filename = filename.strip(" ")[:-4]
    case_id = filename[:11]
    timestamps = filename[12:]
    if timestamps:
        start_timestamp, end_timestamp = timestamps.split("-")
    else:
        start_timestamp, end_timestamp = 0, 0
    return case_id, float(start_timestamp), float(end_timestamp)

def timestamp2frame(timestamp, fps=FPS):
    return int(timestamp * fps)

def add_annotation(eaf, tier_name, data, start_frame, end_frame):
    start_time_list, end_time_list, label_list = parse_data(
        tier_name, data, start_frame, end_frame
        )
    start_time_list = [max(0, elem - start_frame) for elem in start_time_list]
    end_time_list = [elem - start_frame for elem in end_time_list]
    for start_time, end_time, label in zip(
        start_time_list, end_time_list, label_list
        ):
        if start_time == end_time:
            continue
        start_time_ms = int(1000 * start_time / FPS)
        end_time_ms = int(1000 * end_time / FPS)
        eaf.add_annotation(tier_name, start_time_ms, end_time_ms, label)
    return None

def parse_data(tier_name, annotations, start_frame, end_frame):
    annotation = extract_annotation(tier_name, annotations)
    start_time_list = list()
    end_time_list = list()
    annotation_list = list()
    for start_time, end_time, label in zip(*annotation):
        if end_time < start_frame:
            continue
        if start_time > end_frame:
            break
        start_time_list.append(start_time)
        end_time_list.append(min(end_time, end_frame))
        annotation_list.append(label)
    return start_time_list, end_time_list, annotation_list

def extract_annotation(tier_name, annotations):
    axis = get_axis(tier_name)
    zones = extract_zones(tier_name, annotations)
    key = "annotation_" + axis
    return zones[key]

def extract_zones(tier_name, annotations):
    if "Left Wrist" in tier_name:
        return annotations["left_wrist_annotation"][()]
    elif "Right Wrist" in tier_name:
        return annotations["right_wrist_annotation"][()]
    elif "Left Finger" in tier_name:
        return annotations["left_finger_annotation"][()]
    elif "Right Finger" in tier_name:
        return annotations["right_finger_annotation"][()]
    elif "Left Thumb" in tier_name:
        return annotations["left_thumb_annotation"][()]
    elif "Right Thumb" in tier_name:
        return annotations["right_thumb_annotation"][()]

def get_axis(tier_name):
    if "Vertical" in tier_name:
        axis = "y"
    elif "Lateral" in tier_name:
        axis = "x"
    return axis

def extract_timeseries(
        data, timeseries, start_frame, end_frame, *args, **kwargs
        ):
    ret_dict = {"time": np.arange(end_frame - start_frame) / FPS}
    for elem in timeseries:
        elem_data = extract_keypoints_timeseries(
            data, elem, start_frame, end_frame, *args, **kwargs
            )
        ret_dict.update(
            {
            elem + "_horizontal": elem_data[:, 0],
            elem + "_vertical": elem_data[:, 1],
            # elem + "_depth": elem_data[:, 2]
            }
        )
    return ret_dict

def extract_keypoints_timeseries(
        data, keypoint_name, start_frame, end_frame, *args, **kwargs
        ):
    all_scenes = kwargs["annotations"]["all_scenes"][()]
    video_scenes = list()
    for elem in all_scenes:
        if elem[1].frame_num <= start_frame:
            continue
        if elem[0].frame_num >= end_frame:
            break
        video_scenes.append(elem)
    if "right_wrist" in keypoint_name:
        ret_data = data["pose_keypoints_2d_norm"][:, 4, :]
        # ret_data = data["pose_keypoints_2d_norm"][:, 16, :]
    elif "left_wrist" in keypoint_name:
        ret_data = data["pose_keypoints_2d_norm"][:, 7, :]
        # ret_data = data["pose_keypoints_2d_norm"][:, 15, :]
    elif "right_finger" in keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 8, :] \
                    - data["hand_right_keypoints_2d_norm"][:, 6, :]
    elif "right_thumb" in keypoint_name:
        ret_data = data["hand_right_keypoints_2d_norm"][:, 4, :] \
                    - data["hand_right_keypoints_2d_norm"][:, 3, :]
    elif "left_finger" in keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 8, :] \
                    - data["hand_left_keypoints_2d_norm"][:, 6, :]
    elif "left_thumb" in keypoint_name:
        ret_data = data["hand_left_keypoints_2d_norm"][:, 4, :] \
                    - data["hand_left_keypoints_2d_norm"][:, 3, :]
    # elif "right_eyebrow" in keypoint_name:
    #     ret_data = data["face_keypoints_2d"][:, 296, :]
    # elif "left_eyebrow" in keypoint_name:
    #     ret_data = data["face_keypoints_2d"][:, 66, :]
    # elif "right_eyebrow_scene" in keypoint_name:
    #     ret_data = data["face_keypoints_2d_norm"][:, 17:22, :].mean(axis=1)
    # elif "left_eyebrow_scene" in keypoint_name:
    #     ret_data = data["face_keypoints_2d_norm"][:, 22:27, :].mean(axis=1)
    # elif "right_eyebrow_frame" in keypoint_name:
    #     ret_data = data["face_keypoints_2d"][:, 17:22, :].mean(axis=1)
    #     nose = data["face_keypoints_2d"][:, 27, :]
    #     ret_data = np.linalg.norm(ret_data - nose, axis=1)
    #     ret_data = np.stack([ret_data, ret_data]).T
    # elif "left_eyebrow_frame" in keypoint_name:
    #     ret_data = data["face_keypoints_2d"][:, 22:27, :].mean(axis=1)
    #     nose = data["face_keypoints_2d"][:, 27, :]
    #     ret_data = np.linalg.norm(ret_data - nose, axis=1)
    #     ret_data = np.stack([ret_data, ret_data]).T
    elif "right_eyebrow" in keypoint_name:
        data = dict(data)
        for scene in video_scenes:
            scene_start_frame = scene[0].frame_num
            scene_end_frame = scene[1].frame_num
            scene_data = data["face_keypoints_2d_norm"][scene_start_frame:scene_end_frame,...]
            eyebrow_scene_data = scene_data[:, 17:22, :].mean(axis=1)
            eye_scene_data = scene_data[:, 40:42, :].mean(axis=1)
            nu = np.linalg.norm(eyebrow_scene_data-eye_scene_data, axis=1).mean()
            scene_data = (scene_data - eye_scene_data[:, None, :])/(nu + 1e-4)
            data["face_keypoints_2d"][scene_start_frame:scene_end_frame] = scene_data
        ret_data = data["face_keypoints_2d"][:, 17:22, :].mean(axis=1)
    elif "left_eyebrow" in keypoint_name:
        data = dict(data)
        for scene in video_scenes:
            scene_start_frame = scene[0].frame_num
            scene_end_frame = scene[1].frame_num
            scene_data = data["face_keypoints_2d_norm"][scene_start_frame:scene_end_frame,...]
            eyebrow_scene_data = scene_data[:, 22:27, :].mean(axis=1)
            eye_scene_data = scene_data[:, 45:47, :].mean(axis=1)
            nu = np.linalg.norm(eyebrow_scene_data-eye_scene_data, axis=1).mean()
            scene_data = (scene_data - eye_scene_data[:, None, :])/(nu + 1e-4)
            data["face_keypoints_2d"][scene_start_frame:scene_end_frame] = scene_data
        ret_data = data["face_keypoints_2d"][:, 22:27, :].mean(axis=1)
    else:
        raise ValueError
    return ret_data[start_frame:end_frame]
        
