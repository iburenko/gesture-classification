from os import listdir as ls, path

from scenedetect import SceneManager, ContentDetector

def find_scenes(video, threshold=27.0):
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()

def find_video_file(elan_file, video_home):
    filename = path.basename(elan_file)
    filename = filename.split(")")
    if len(filename) == 1:
        filename = filename[0]
    else:
        filename = filename[1]
    filename = filename.strip(" ")[:-4]
    all_video_files = ls(video_home)
    video_file = [elem for elem in all_video_files if filename in elem]
    if video_file:
        return video_file[0]
    else:
        return None
