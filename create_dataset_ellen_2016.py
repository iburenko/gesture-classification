from os import path, makedirs

import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm

FPS = 29.97

# parser = argparse.ArgumentParser()
# parser.add_argument('--tmpdir-home', type=str)
# parser.add_argument('--split-home', type=str)
# parser.add_argument('--csv-home', type=str)
# parser.add_argument('--videos-home', type=str)
# parser.add_argument('--length-msec', type=int)
# parser.add_argument('--threshold', type=float)
# args = parser.parse_args()

original_shows_path = (
    "/home/atuin/b105dc/data/datasets/"
    "ellen_show_datasets/all_ellen_shows_dates.txt"
    )
with open(original_shows_path, "r") as file:
    original_shows_dates = file.readlines()
original_shows_dates = [elem.strip()[1:-1] for elem in original_shows_dates]

class args:
    tmpdir_home = "/scratch/b105dc10"
    data_path = (
        "/home/atuin/b105dc/data/datasets/"
        "ellen_show_datasets/ellen_degeneres_2016_all_video_paths.txt"
    )
    length_msec = 10000

with open(args.data_path, 'r') as file:
    data = file.readlines()
data = [elem.strip() for elem in data]

original_shows_list = list()
for elem in data:
    original_show = any(show_date in elem for show_date in original_shows_dates)
    if original_show:
        original_shows_list.append(elem)

node_path = args.tmpdir_home
length_msec = args.length_msec

home_path = path.join(node_path, "ellen_degeneres_2016_all_data_300_frames")
makedirs(path.join(home_path, "video"), exist_ok=True)
makedirs(path.join(home_path, "audio"), exist_ok=True)

def create_dataset(data):
    clip_length = length_msec/1000
    file_pbar = tqdm(data, total=len(data))
    for fp in file_pbar:
        video_obj = VideoFileClip(fp)
        video_duration = np.floor(video_obj.duration).astype('int')
        i = 0
        start_sec = 0
        fn = path.basename(fp).split('.')[0]
        file_pbar.set_description(
            f"Processing file {fn}"
        )
        num_snippets = int(video_duration // clip_length)
        for i in tqdm(range(num_snippets), leave=False):
            start_sec, end_sec = i * clip_length, (i + 1) * clip_length
            subclip = video_obj.subclip(start_sec, end_sec)            
            suffix = f"{start_sec:.1f}_{end_sec:.1f}"
            snippet_fn = fn + "_" + suffix
            audio_filename = path.join(home_path, "audio", snippet_fn)
            video_filename = path.join(home_path, "video", snippet_fn)
            subclip.write_videofile(video_filename + ".mp4", audio=False, verbose=False, logger=None)
            subclip.audio.write_audiofile(audio_filename + ".wav", fps=16000, verbose=False, logger=None)

if __name__ == "__main__":
    create_dataset(original_shows_list)