# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
from os.path import join
import argparse
import numpy as np
from tqdm import tqdm
import soundfile as sf
import matplotlib.pyplot as plt
import subprocess
import librosa
from pathlib import Path
from PIL import Image
import cv2
import shutil

def write_video_with_audio(save_root, audio_path, output_path, h=512, w=512, fps=25):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2
    text = 'This video has been manipulated.'

    label_width, label_height = cv2.getTextSize(text, font, 1, 2)[0]
    print(label_width, label_height)
    bottomLeftCornerOfText = (int((w - label_width) / 2), int(h - label_height - 20))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_tmp_path = os.path.join(save_root, 'tmp.avi')
    num_images = int(len(os.listdir(save_root)))

    out = cv2.VideoWriter(video_tmp_path, fourcc, fps, (w, h))
    for j in tqdm(range(num_images), position=0, desc='writing video'):
        img = cv2.imread(os.path.join(save_root, '%05d.png' % j))
        img = cv2.putText(
            img,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        out.write(img)

    out.release()

    print("ffmpeg version:")
    subprocess.call("ffmpeg -version", shell=True)

    # TODO: Set proper size of video: [-s {w}x{h}]
    cmd = f'ffmpeg -y -i "{video_tmp_path}" -i "{audio_path}" -vcodec libx264 -c:a aac "{output_path}"'
    # "-pix_fmt yuv420p -profile:v baseline -level 3"

    print(f"ffmpeg cmd: {cmd}")
    return_code = subprocess.call(cmd, shell=True)

    if return_code > 0:
        raise Exception(f"An error occurred when assembling the output video: ffmpeg return_code={return_code}")

    # try:
    #     os.remove(video_tmp_path)  # remove the template video

    # except FileNotFoundError:
    #     return

def write_video_with_audio_old(save_root, audio_path, output_path, h=512, w=512, fps=25):

    cmd = 'ffmpeg -y -r ' + str(fps) + ' -s ' + f'{w}x{h}' + ' -i "' + f'{save_root}/%05d.png' + \
        '" -i "' + audio_path + '" -vcodec libx264 -c:a aac "' + output_path + '"'
    subprocess.call(cmd, shell=True)
    

if __name__ == '__main__':

    # load args 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--name_audio', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--fps', required=True, type=np.int32)
    parser.add_argument('--sr', required=True, type=np.int32)
    parser.add_argument('--all_vids', required=False, action='store_true', help='Generate all videos?')
    parser.add_argument('--enhance', required=False, action='store_true', help='Enhance generated video. (Takes longer).')
    parser.add_argument('--clean', required=False, action='store_true', help='Delete everything but the videos?')
    parser.add_argument('--move_to_one_folder', required=False, action='store_true', help='Put generated video into videos folder.')

    inopt = parser.parse_args()

    # make videos
    # generate corresponding audio, reused for all results
    audio_path = os.path.join(inopt.dataroot, 'audio', inopt.name_audio, inopt.name_audio + '.wav')
    audio, _ = librosa.load(audio_path, sr=inopt.sr)

    # make generated video
    generated_frames_path = os.path.join(inopt.out_dir, 'generated_frames')
    edges_path = os.path.join(inopt.out_dir, 'edges')
    render_path = os.path.join(inopt.out_dir, 'render')

    try:
        nframe = len(os.listdir(generated_frames_path))
        w, h = Image.open(os.path.join(generated_frames_path, os.listdir(generated_frames_path)[0])).size

    except (IndexError, FileNotFoundError) as e:
        nframe = len(os.listdir(render_path))
        w, h = Image.open(os.path.join(render_path, os.listdir(render_path)[0])).size
    
    print(f'Image size: height {h} width {w}')

    tmp_audio_path = os.path.join(inopt.out_dir, 'tmp.wav')
    tmp_audio_clip = audio[: np.int32(nframe * inopt.sr / inopt.fps)]
    sf.write(tmp_audio_path, tmp_audio_clip, inopt.sr)

    # TODO: when deploying make one or the other
    final_video_path = os.path.join(inopt.out_dir, 'generated_video.mp4')
    write_video_with_audio(generated_frames_path, tmp_audio_path, final_video_path, h, w, inopt.fps)

    if inopt.enhance:
        superes_frames_path = os.path.join(inopt.out_dir, 'superes/generated_frames')
        w, h = Image.open(os.path.join(superes_frames_path, os.listdir(superes_frames_path)[0])).size
        # make video of edges for comparison
        video_superes_path = os.path.join(inopt.out_dir, 'generated_superes.mp4')
        write_video_with_audio(superes_frames_path, tmp_audio_path, video_superes_path, h, w, inopt.fps)   

    if inopt.all_vids:
        # make video of edges for comparison
        video_edges_path = os.path.join(inopt.out_dir, 'generated_edges.mp4')
        write_video_with_audio(edges_path, tmp_audio_path, video_edges_path, h, w, inopt.fps)

        # make video of render for comparison
        video_render_path = os.path.join(inopt.out_dir, 'generated_render.mp4')
        write_video_with_audio(render_path, tmp_audio_path, video_render_path, h, w, inopt.fps)

        # make side to side video: render 2 edges
        combined_path = os.path.join(inopt.out_dir, 'render2edges.mp4')
        cmd = 'ffmpeg -y -i "' + video_render_path + '" -i "' + \
            video_edges_path + '" -filter_complex hstack -codec:v libx264 -crf 0 -preset veryslow "' + combined_path + '"'
        subprocess.call(cmd, shell=True)

        # make side to side video: edges 2 video
        combined_path = os.path.join(inopt.out_dir, 'edges2video.mp4')
        cmd = 'ffmpeg -y -i "' + video_edges_path + '" -i "' + \
            final_video_path + '" -filter_complex hstack -codec:v libx264 -crf 0 -preset veryslow "' + combined_path + '"'
        subprocess.call(cmd, shell=True)

        # make side to side video: render 2 edges 2 video
        combined_path = os.path.join(inopt.out_dir, 'render2edges2video.mp4')

        cmd = 'ffmpeg -y -i "' + video_render_path + '" -i "' + \
            video_edges_path + '" -i "' + final_video_path + \
            '" -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" -map "[v]" -map 1:a -codec:v libx264 -crf 0 -preset veryslow "' + combined_path + '"'
        
        subprocess.call(cmd, shell=True)

    if os.path.exists(tmp_audio_path):
        os.remove(tmp_audio_path)
    
    #TODO: move generated video to /videos/audio_to_video.mp4
    if inopt.move_to_one_folder:
        parent, name = os.path.split(inopt.out_dir)
        shutil.move(final_video_path, os.path.join(parent, 'videos', name + '.mp4'))

        if inopt.clean:
            shutil.rmtree(inopt.out_dir)

    # delete subfolder
    else:
        if inopt.clean:
            # clean all subfolders
            subfolders = [f.path for f in os.scandir(inopt.out_dir) if f.is_dir()]
            for sub in subfolders:
                shutil.rmtree(sub)
            
            # remove headposes
            try:
                os.remove(os.path.join(inopt.out_dir, 'headposes.npy')) 
        
            except FileNotFoundError:
                pass
    
    
    print('Finish!')
