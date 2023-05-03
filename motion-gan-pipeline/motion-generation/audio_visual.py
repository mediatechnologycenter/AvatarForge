# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
from os.path import join
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
import sys
sys.path.append('../preprocessing/')
from face_tracking.FLAME.FLAME import FLAME
from face_tracking.FLAME.config import cfg as FLAME_cfg
from face_tracking.FLAME.lbs import vertices2landmarks
from face_tracking.render_3dmm import Render_FLAME
from face_tracking.util import *
import torch
from scipy.ndimage import gaussian_filter1d

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
    target_dir= './visuals/Audio_Visual/'
    os.makedirs(target_dir, exist_ok=True)

    # Make renderer
    h_fl, w_fl = 512, 512
    focal = torch.from_numpy(np.array([900.], dtype=np.float32))

    cxy = torch.tensor((w_fl / 2.0, h_fl / 2.0), dtype=torch.float).cpu()

    model_3dmm = FLAME(FLAME_cfg.model)
    renderer = Render_FLAME(model_3dmm.faces_tensor, focal, h_fl, w_fl, 1, device)

    stop_at = 50
    
    for file_index in tqdm(range(len(dataset))):

        if file_index >= stop_at:
            break

        test_name = dataset.clip_names[file_index]

        name, flame, frames, audio_packed, audio_features, emotions, landmarks, exps, headposes, velocity_pose, fit_data_path = dataset[file_index]

        # Spectrogram of frequency
        audio, sr = audio_packed
        fps = 25
        audio_len = len(audio)/sr

        # load fit data
        fit_data = torch.load(fit_data_path)

        # Amplitude
        X = audio
        time_sf = np.linspace(start=0, stop=len(X)/sr, num=len(X))
        lim = np.max([np.abs(np.min(X)), np.max(X)])

        # Spectogram
        X = librosa.stft(audio)
        Xdb = librosa.amplitude_to_db(abs(X))

        # RMS
        S, phase = librosa.magphase(X)
        rms = librosa.feature.rms(S=S)
        times = librosa.times_like(rms)

        # log Power spectrogram
        lpw = librosa.amplitude_to_db(S, ref=np.max)
        
        # Zero Crossing Rate ZCR
        zcrs = librosa.feature.zero_crossing_rate(audio)

        # Mel-Frequency Cepstral Coefficients (MFCC)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr)

        # Make frames for each fps
        for i in tqdm(range(min(len(frames), len(audio_features)))):

            line_at = i/fps
            
            fig = plt.figure(figsize=(16, 15), tight_layout=True)
            fontsize = 18
            gs = GridSpec(5, 2)
            current_plot = 0
            column = 1

            # Amplitude
            ax1 = fig.add_subplot(gs[current_plot, column])
            ax1.set_title('Amplitude', fontsize=fontsize-2)
            ax1.set_ylabel('Amplitude')
            ax1.set_xlabel('Time')
            ax1.plot(time_sf, audio, label=name, alpha=0.5)
            ax1.set_xlim([0, audio_len])
            ax1.set_ylim([-lim, lim])
            ax1.axvline(line_at, color = 'r')
            current_plot += 1

            # Spectogram
            ax2 = fig.add_subplot(gs[current_plot, column])
            ax2.set_title('Spectogram', fontsize=fontsize-2)
            librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
            # plt.colorbar(orientation="horizontal", ax=ax2)
            ax2.axvline(line_at, color = 'r')
            current_plot += 1

            # RMS
            ax3 = fig.add_subplot(gs[current_plot, column])
            ax3.set_title('RMS - Total magnitude / energy', fontsize=fontsize)
            ax3.semilogy(times, rms[0], label='RMS Energy')
            ax3.set_xlim([0, audio_len])
            ax3.set_ylim([0, np.max(rms[0])])
            plt.axvline(line_at, color = 'r')
            current_plot += 1

            # log Power spectrogram
            ax4 = fig.add_subplot(gs[current_plot, column])
            ax4.set_title('Log Power Spectrogram', fontsize=fontsize-2)
            librosa.display.specshow(lpw, y_axis='log', x_axis='time', ax=ax4)
            # plt.colorbar(orientation="horizontal", ax=ax4)
            plt.axvline(line_at, color = 'r')
            current_plot += 1

            # Mel-Frequency Cepstral Coefficients (MFCC)
            ax5 = fig.add_subplot(gs[current_plot, column])
            ax5.set_title('Mel-Frequency Cepstral Coefficients (MFCC)', fontsize=fontsize-2)
            librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax5)
            # plt.colorbar(orientation="horizontal", ax=ax5)
            plt.axvline(line_at, color = 'r')
            current_plot += 1

            # Right column
            column = 0

            # Original frame
            ax6 = fig.add_subplot(gs[0:2, column])
            ax6.set_title(name, fontsize=fontsize + 2)
            ax6.imshow(Image.open(frames[i]))
            plt.axis('off')

            # Talking head
            R = fit_data['euler'][i]
            R = R.unsqueeze(0).to(device).double()
            t = torch.from_numpy(np.array([0, 0, -5])).unsqueeze(0).to(device).double()
            id = fit_data['id'].to(device).double()
            expr = exps[i].unsqueeze(0).to(device).double()
            rott_geo = model_3dmm.forward_geo(id, expr, R, t)
            landmarks3d = model_3dmm.get_3dlandmarks(id, expr, R, t, focal, cxy).cpu()
            proj_geo = proj_pts(landmarks3d, focal, cxy)
            render_imgs = renderer(rott_geo.float(), model_3dmm.faces_tensor)
            img_arr = render_imgs[0, :, :, :3].cpu().numpy()
            img_arr *= 255
            img_arr = img_arr.astype(np.uint8)
            im = Image.fromarray(img_arr)
            ax7 = fig.add_subplot(gs[2:4, column])
            ax7.set_title('Flame Head', fontsize=fontsize)
            ax7.imshow(im)
            # ax7.imshow(Image.open(flame[i]))
            plt.axis('off')

            # Deepspeech visuals
            ax8 = fig.add_subplot(gs[4, column])
            deepspeech = audio_features[i].reshape((16,29))
            ax8.set_title('Deepspeech Features', fontsize=fontsize-2)
            ax8.imshow(deepspeech, cmap='hot', interpolation='nearest')

            # Save and close
            plt.savefig(os.path.join(target_dir, f'audio_{i}.png'))
            plt.close()

        # make videos
        final_path = os.path.join(target_dir, test_name + '.avi')
        audio_path = os.path.join(dataset.dataset_root, test_name, test_name +'.wav')
        tmp = Image.open(os.path.join(target_dir, f'audio_{i}.png'))
        w, h = tmp.size
        write_video_with_audio(audio_path, final_path, 'audio_', h, w, fps)

        _img_paths = list(map(lambda x:str(x), list(Path(target_dir).glob('*.png'))))
        for i in tqdm(range(len(_img_paths)), desc='deleting intermediate images'):
            os.remove(_img_paths[i])

    print('Finish!')
