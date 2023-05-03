# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
from moviepy.editor import *

dataset = 'SRF_anchor'
data_path = f'/home/alberto/data/videosynth/{dataset}'
target_base = f'/home/alberto/data/videosynth/{dataset}_frames'

for sub in ['Close', 'Halbtotale', 'Totale']:
    os.makedirs(os.path.join(target_base, sub), exist_ok=True)

video_list = ['Close/355_9105.mp4', 'Close/355_9106.mp4',
              'Halbtotale/355_9414.mp4', 'Halbtotale/355_9415.mp4',
              'Totale/355_7615.mp4', 'Totale/355_7616.mp4']

# Start and end time in minutes
frame_n = 200

for video in video_list:
    input_file = os.path.join(data_path, video)
    output_file = os.path.join(target_base, video)
    # print(os.path.join(target_base, video))
    # ffmpeg_extract_subclip(os.path.join(data_path, video), start_time*60, end_time*60,
    #                        targetname=os.path.join(target_base, video))

    clip = VideoFileClip(input_file)
    clip.save_frame(f'{target_base}/{video[:-4]}.jpeg', t=frame_n)  # save frame at t=2 as JPEG
