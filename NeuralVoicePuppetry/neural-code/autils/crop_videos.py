# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
from moviepy.editor import *
from subprocess import call
from math import ceil

dataset = 'External'
data_path = f'/home/alberto/data/videosynth/{dataset}'
target_base = f'/home/alberto/data/videosynth/{dataset}'

for sub in ['Close', 'Halbtotale', 'Totale']:
    os.makedirs(os.path.join(target_base, sub), exist_ok=True)

video_list = ['Youtube/Russian_guy.mp4', ]

# Start and end time in minutes
start_time = 0
end_time = 5

for video in video_list:
    input_file = os.path.join(data_path, video)
    output_file = os.path.join(target_base, video)

    clip = VideoFileClip(input_file)
    print('FPS: ', clip.fps)
    # end_time = ceil((7500/clip.fps))/60
    # print('End time: ', end_time)

    try:
        cut = clip.subclip(start_time*60, int(end_time*60))

        cut.write_videofile(output_file)

    except ValueError:
        start_time = 0
        end_time = 5

        cut = clip.subclip(start_time * 60, int(end_time*60))
        cut.write_videofile(output_file)

