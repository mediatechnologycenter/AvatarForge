# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import numpy as np
import pandas as pd
import imageio
import os
import subprocess
from multiprocessing import Pool
from itertools import cycle
import warnings
import glob
import time
from tqdm import tqdm
from util import save
from argparse import ArgumentParser
from skimage import img_as_ubyte
from skimage.transform import resize
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.fx.all import crop
from moviepy.editor import *

warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')

def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path

def run(data):
    video_id, args = data
    
    if not os.path.exists(os.path.join(args.video_folder, video_id + '.mp4')):
       download(video_id, args)

    if not os.path.exists(os.path.join(args.video_folder, video_id + '.mp4')):
        print('Broken Link: ', video_id)
        return

    if args.mode == 'd':
        return
    try:
        reader = imageio.get_reader(os.path.join(args.video_folder, video_id + '.mp4'))

    except FileNotFoundError:
        print(f'Could not find: {os.path.join(args.video_folder, video_id + ".mp4")}')
        return
    
    frame = reader.get_next_data()
    fps = reader.get_meta_data()['fps']
    df = pd.read_csv(args.metadata)
    df = df[df['video_id'] == video_id]

    all_chunks_dict = [{'start': df['start'].iloc[j], 'end': df['end'].iloc[j],
                        'bbox': list(map(int, df['bbox'].iloc[j].split('-'))), 'frames': []} for j in
                       range(df.shape[0])]

    ref_height = df['height'].iloc[0]
    ref_width = df['width'].iloc[0]
    partition = df['partition'].iloc[0]

    for entry in all_chunks_dict:

        if 'person_id' in df:
            first_part = df['person_id'].iloc[0] + "#"
        else:
            first_part = ""
        first_part = first_part + '#' + video_id
        inpath = os.path.join(args.video_folder, video_id + '.mp4')
        path = first_part + '#' + str(entry['start']).zfill(6) + '#' + str(entry['end']).zfill(6) + '.mp4'
        outpath = os.path.join(args.out_folder, partition, path)

        clip = VideoFileClip(inpath)
        print('start: ', entry['start']/fps, ', end: ', entry['end']/fps)

        try:
            clip = clip.subclip(entry['start']/fps, entry['end']/fps)
            
        except ValueError as e:
            print(e)
            return

        left, top, right, bot = entry['bbox']

        left = int(left / (ref_width / frame.shape[1]))
        top = int(top / (ref_height / frame.shape[0]))
        right = int(right / (ref_width / frame.shape[1]))
        bot = int(bot / (ref_height / frame.shape[0]))
        clip = crop(clip, left, top, right, bot)

        # saving the clip
        clip.write_videofile(outpath)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--video_folder", help='Path to youtube videos')
    parser.add_argument("--metadata", help='Path to metadata')
    parser.add_argument("--out_folder", help='Path to output')
    parser.add_argument("--format", help='Storing format')
    parser.add_argument("--workers", default=1, type=int, help='Number of workers')
    parser.add_argument("--youtube", default='./youtube-dl', help='Path to youtube-dl')
    parser.add_argument("--mode", default='d_p', help='Mode: download (d) or download and process (d_p)')

    args = parser.parse_args()
    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    df = pd.read_csv(args.metadata)
    video_ids = df['video_id']

    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for chunks_data in tqdm(pool.imap_unordered(run, zip(video_ids, args_list))):
        None


