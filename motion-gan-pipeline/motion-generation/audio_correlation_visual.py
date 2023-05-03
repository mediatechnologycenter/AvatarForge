# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
from os.path import join
from shutil import move
import sys
from turtle import shape
import yaml
import argparse
import numpy as np
from options.test_audio2headpose_options import TestOptions
from datasets import create_dataset
from util.cfgnode import CfgNode
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import librosa
import librosa.display
from PIL import Image
sys.path.append('../preprocessing/')
from face_tracking.FLAME.FLAME import FLAME
from face_tracking.FLAME.config import cfg as FLAME_cfg
from face_tracking.FLAME.lbs import vertices2landmarks
from face_tracking.render_3dmm import Render_FLAME
from face_tracking.util import *
import torch
from scipy.stats import pearsonr


def write_video_with_audio(audio_path, output_path, prefix='pred_', h=512, w=512, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_tmp_path = join(target_dir, 'tmp.avi')
    out = cv2.VideoWriter(video_tmp_path, fourcc, fps, (w, h))
    nframe = len(list(Path(target_dir).glob('*.png')))

    for j in tqdm(range(nframe), position=0, desc='writing video'):
        img = cv2.imread(join(target_dir, prefix + str(j+1) + '.png'))
        out.write(img)
    out.release()
    cmd = 'ffmpeg -y -i "' + video_tmp_path + '" -i "' + \
        audio_path + '" -codec copy -shortest "' + output_path + '"'
    subprocess.call(cmd, shell=True)
    os.remove(video_tmp_path)  # remove the template video


if __name__ == '__main__':
    # Load default options
    Test_parser = TestOptions()
    opt = Test_parser.parse()   # get training options

    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Audio2Headpose_Ted.yml',
                        help="person name, e.g. Obama1, Obama2, May, Nadella, McStay", required=False)
    inopt = parser.parse_args()
    # TODO: make config
    with open(inopt.config) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Overwrite with config
    opt.phase = 'Test'
    opt.name = cfg.experiment.name
    opt.dataset_mode = cfg.experiment.dataset_mode
    opt.dataroot = cfg.experiment.dataroot
    opt.dataset_names = cfg.experiment.dataset_names
    opt.FPS = cfg.experiment.fps

    # save to the disk
    Test_parser.print_options(opt)

    # Set device
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if len(
            opt.gpu_ids) > 0 else torch.device('cpu')

    # Load data
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    dataset = dataset.dataset

    # Target dir
    target_dir= './visuals/Audio_Correlation/'
    os.makedirs(target_dir, exist_ok=True)

    # Make renderer
    h_fl, w_fl = 512, 512
    focal = torch.from_numpy(np.array([900.])).double()

    cxy = torch.tensor((w_fl / 2.0, h_fl / 2.0), dtype=torch.float).cpu()

    model_3dmm = FLAME(FLAME_cfg.model)
    renderer = Render_FLAME(model_3dmm.faces_tensor, focal, h_fl, w_fl, 1, device)

    stop_at = 50

    movement = []
    amplitude = []
    
    for file_index in tqdm(range(len(dataset))):

        if file_index >= stop_at:
            break

        test_name = dataset.clip_names[file_index]

        name, flame, frames, audio_packed, audio_features, emotions, landmarks, exps, headposes, velocity_pose, fit_data_path = dataset[file_index]

        nframes = headposes.shape[0]

        # Spectrogram of frequency
        audio, sr = audio_packed
        fps = 25
        audio_len = len(audio)/sr

        # load fit data
        fit_data = torch.load(fit_data_path)

        # Spectogram
        X = librosa.stft(audio)
        # print('Amplitude shape: ', audio.shape)
        # print('Num frames: ', nframes)

        # RMS
        S, phase = librosa.magphase(X)
        rms = librosa.feature.rms(S=S)

        ampl_delta = np.concatenate([[0], audio[1:] - audio[:-1]])

        interval = len(audio)/nframes

        # Over teh spast second
        window = fps * 2
        for frame in range(window, nframes):
            mov = np.sum(np.abs(velocity_pose[frame-window:frame]), axis=0)
            # mov = velocity_pose[frame]
            # mov = headposes[frame]

            ampl = np.sum(np.abs(ampl_delta[int((frame-window)*interval):int(frame*interval)]))
            # ampl = np.sum((ampl_delta[int((frame-window)*interval):int(frame*interval)]))
            # ampl = np.mean(audio[int((frame-window)*interval):int(frame*interval)])
            movement.append(mov)
            amplitude.append(ampl)
    
    movement = np.array(movement)
    amplitude = np.array(amplitude)

    print(movement.shape)
    print(amplitude.shape)

    for idx, angle in enumerate(['Pitch', 'Yaw', 'Roll']):
        
        # Movement per frame
        r, p = pearsonr(x=amplitude, y=movement[:, idx])
        r = round(r, 3)
        p = round(p, 3)
        x = np.linspace(np.min(amplitude), np.max(amplitude), 1000)

        plt.title(f'{angle} Angle total movement and total change in Amplitude in the past 2 seconds \n Correlation: {int(r*100)}%, p-value: {int(p*100)}%')
        plt.scatter(x=amplitude, y=movement[:, idx])  

        plt.plot(x, (x * r) + np.mean(movement[:, idx]), linestyle='solid', color='red')
        plt.xlabel('Amplitude')
        plt.ylabel('Movement')
        plt.savefig(f'{target_dir}{angle}_movement&delta_Amplitude.png') 
        plt.close()

        # Talking head
        R = torch.zeros(3)
        R[idx] += 45
        R = R.unsqueeze(0).to(device).double()
        t = torch.from_numpy(np.array([0, 0, -5])).unsqueeze(0).to(device).double()
        id = torch.zeros_like(fit_data['id']).to(device).double()
        expr = torch.zeros_like(exps[0]).unsqueeze(0).to(device).double()

        rott_geo = model_3dmm.forward_geo(id, expr, R, t)
        landmarks3d = model_3dmm.get_3dlandmarks(id, expr, R, t, focal, cxy).cpu()
        proj_geo = proj_pts(landmarks3d, focal, cxy)

        sel_pts3D = vertices2landmarks(rott_geo,
                                    model_3dmm.faces_tensor,
                                    model_3dmm.full_lmk_faces_idx.repeat(
                                        1, 1),
                                    model_3dmm.full_lmk_bary_coords.repeat(1, 1, 1))

        render_imgs = renderer(rott_geo.float(), model_3dmm.faces_tensor)
        img_arr = render_imgs[0, :, :, :3].cpu().numpy()
        img_arr *= 255
        img_arr = img_arr.astype(np.uint8)
        im = Image.fromarray(img_arr)
        plt.imshow(im)
        plt.savefig(f'{target_dir}{angle}_45.png')
        plt.close()

    print('Finish!')
