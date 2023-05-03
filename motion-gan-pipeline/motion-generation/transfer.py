# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from cmath import inf
import os
from os.path import join
import yaml
import argparse
import numpy as np
from options.test_audio2headpose_options import TestOptions
from datasets import create_dataset
from models import create_model
from util.cfgnode import CfgNode
import cv2
import librosa
import soundfile as sf
from tqdm import tqdm
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path
import torch
from funcs import utils
import sys
sys.path.append('../preprocessing/')
from face_tracking.FLAME.FLAME import FLAME
from face_tracking.FLAME.config import cfg as FLAME_cfg
from face_tracking.FLAME.lbs import vertices2landmarks
from face_tracking.render_3dmm import Render_FLAME
from face_tracking.util import *
from finetune import finetune
from PIL import Image


def write_video_with_audio(audio_path, output_path, prefix='pred_', h=512, w=512, fps=25):
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_tmp_path = join(save_root, 'tmp.avi')
    out = cv2.VideoWriter(video_tmp_path, fourcc, fps, (w, h))
    for j in tqdm(range(nframe), position=0, desc='writing video'):
        img = cv2.imread(join(save_root, prefix + str(j+1) + '.jpg'))
        out.write(img)
    out.release()
    cmd = 'ffmpeg -y -i "' + video_tmp_path + '" -i "' + \
        audio_path + '" -codec copy -shortest "' + output_path + '"'
    subprocess.call(cmd, shell=True)

    os.remove(video_tmp_path)  # remove the template video



if __name__ == '__main__':

    # load args 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True)
    parser.add_argument('--dataset_names', required=True)
    parser.add_argument('--target_name', required=True)
    parser.add_argument('--checkpoint_dir', required=True)
    parser.add_argument('--out_dir', required=True)

    inopt = parser.parse_args()

    # Load default options
    Test_parser = TestOptions()
    opt = Test_parser.parse()   # get training options

    # Overwrite with config
    opt.phase = ''
    opt.dataset_mode = 'audio'  # for testing
    opt.dataroot = os.path.join(inopt.dataroot, 'audio')
    opt.dataset_names = inopt.dataset_names
    opt.FPS = 25
    opt.serial_batches = True
    opt.train_test_split = False

    # save to the disk
    Test_parser.print_options(opt)

    # Set device
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if len(
             opt.gpu_ids) > 0 else torch.device('cpu')

    # Load Model
    print('---------- Loading Model: {} -------------'.format(opt.task))
    checkpoint_path = os.path.join(inopt.checkpoint_dir, inopt.target_name, 'latest_Audio2Headpose.pkl')
    print(checkpoint_path)
    if not os.path.isfile(checkpoint_path):
        print('No fine-tuned checkpoint for headposes found..')
        finetune(name=inopt.target_name, 
                 dataroot=inopt.dataroot, 
                 dataset_names=inopt.target_name, 
                 target_checkpoints=inopt.checkpoint_dir, 
                 checkpoint_path=os.path.join(inopt.checkpoint_dir, 'Audio2Headpose_TED_checkpoint.pkl'),
                 fps=25)

    Audio2Headpose = create_model(opt)
    Audio2Headpose.load_checkpoint(checkpoint_path)
    Audio2Headpose.eval()

    # Load data
    dataset = create_dataset(opt)
    fit_data_path = os.path.join(inopt.dataroot, 'video', inopt.target_name, 'track_params.pt')
    fit_data = torch.load(fit_data_path)

    stop_at = inf

    for iter, file_index in enumerate(dataset.dataset.valid_clips):

        if iter >= stop_at:
            break

        test_name = dataset.dataset.clip_names[dataset.dataset.valid_clips[file_index]]
        #____________________________________________________#
        print('Generating movement for video: ', test_name)

        # Get features
        audio_features = dataset.dataset.audio_features[file_index]
        # audio_expr = torch.from_numpy(np.stack([np.load(os.path.join(inopt.out_dir,'audio_expr', ae)).flatten() for ae in os.listdir(os.path.join(inopt.out_dir,'audio_expr'))], axis=0))
        # audio_expr = torch.from_numpy(np.stack([np.load(os.path.join(inopt.dataroot, 'video', inopt.target_name,'deca_expr', ae)).flatten() for ae in sorted(os.listdir(os.path.join(inopt.dataroot, 'video', inopt.target_name,'deca_expr')))], axis=0))
        
        # Audio2Headpose
        print('Headpose inference...')

        # Set history headposes as first tracked headposes
        init_rot = fit_data['euler'][0].numpy()
        init_trans = fit_data['trans'][0].numpy() - fit_data['trans'].mean(axis=0).numpy()
        pre_headpose = np.concatenate([init_rot, init_trans], axis=0)
        pre_headpose = np.concatenate([pre_headpose, np.zeros_like(pre_headpose)], axis=0) # headpose and velocity: velocity is zero: still head.

        # Set history headposes as zeros
        # pre_headpose = np.zeros(opt.A2H_wavenet_input_channels, np.float32)

        print('Initial Headpose for prediction: \n', pre_headpose)

        # Generate headposes
        pred_Head = Audio2Headpose.generate_sequences(audio_features, pre_headpose, fill_zero=True, sigma_scale=0.3, opt=opt)

        # Build FLAME and Renderer
        h = fit_data['h']
        w = fit_data['w']
        focal = fit_data['focal']
        id_para = fit_data['id']
        # expr_para = audio_expr
        # nframe = min(expr_para.size()[0], pred_Head.shape[0])

        nframe = pred_Head.shape[0]


        cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).cpu()

        model_3dmm = FLAME(FLAME_cfg.model)
        renderer = Render_FLAME(model_3dmm.faces_tensor, focal, h, w, 1, device)

        # Smooth predicted headpose
        Head_smooth_sigma = [5, 10]
        pred_headpose = utils.headpose_smooth(pred_Head[:, :6], Head_smooth_sigma, method='gaussian').astype(np.float32)
        # Head_smooth_sigma = [10, 25]
        # pred_headpose = utils.headpose_smooth(pred_Head[:, :6], Head_smooth_sigma , method='median').astype(np.float32)

        # Postprocessing
        if fit_data:
            
            og_rot = fit_data['euler'].numpy()
            og_trans = fit_data['trans'].numpy()

            mean_translation = og_trans.mean(axis=0)

            pred_headpose[:, 3:] += mean_translation
            # pred_headpose[:, 0] += 180

        # Make images
        save_root = inopt.out_dir
        os.makedirs(os.path.join(save_root, 'render'), exist_ok=True)
        os.makedirs(os.path.join(save_root, 'landmarks'), exist_ok=True)
        
        np.save(os.path.join(save_root, 'headposes.npy'), pred_headpose)

        deca_expr_path = os.path.join(inopt.dataroot, 'video', inopt.target_name,'deca_expr')
        audio_expr_path = os.path.join(inopt.out_dir, 'audio_expr')

        for i in tqdm(range(nframe), desc='rendering: '):
            R = torch.from_numpy(pred_headpose[i, 0:3]).unsqueeze(
                0).to(device).double()
            t = torch.from_numpy(pred_headpose[i, 3:6]).unsqueeze(
                0).to(device).double()
            
            # Zero translation
            # t = torch.tensor((0, 0, -5)).unsqueeze(
            #     0).to(device).double()

            id = id_para.to(device).double()
            expr = torch.from_numpy(np.load(os.path.join(audio_expr_path, '%05d.npy' % i))).to(device).double()

            '''
            # Original Rotation and Translation used for debug
            if 0:
                og_R = torch.from_numpy(og_rot[i]).unsqueeze(0).to(device).double()
                og_t = torch.from_numpy(og_trans[i]).unsqueeze(
                    0).to(device).double()
                print(
                    f'OG Rotation euler: \n{og_R.data}, \nOG trans: \n{og_t.data}')
                print(
                    f'\nPred Rotation euler: \n{R.data},\nPred trans: \n{t.data}')

                og_rott_geo = model_3dmm.forward_geo(id, expr, og_R, og_t)

                og_landmarks3d = model_3dmm.get_3dlandmarks(
                    id, expr, og_R, og_t, focal, cxy).cpu()
                og_proj_geo = proj_pts(og_landmarks3d, focal, cxy)
                # Porj points
                colormap_blue = plt.cm.Blues
                for num, lin in enumerate(np.linspace(0, 0.9, len(og_proj_geo[0, :, 0]))):
                    plt.scatter(og_proj_geo[0, num, 0].detach().cpu(),
                                og_proj_geo[0, num, 1].detach().cpu(),
                                color=colormap_blue(lin),
                                s=10)
            
            '''

            rott_geo = model_3dmm.forward_geo(id, expr, R, t)
            landmarks3d = model_3dmm.get_3dlandmarks_forehead(id, expr, R, t, focal, cxy).cpu()
            proj_geo = proj_pts(landmarks3d, focal, cxy)
            np.savetxt(os.path.join(save_root, 'landmarks', '%05d.lms' % i), proj_geo[0].detach().cpu().numpy())
            render_imgs = renderer(rott_geo.float(), model_3dmm.faces_tensor)
            img_arr = render_imgs[0, :, :, :3].cpu().numpy()
            img_arr *= 255
            img_arr = img_arr.astype(np.uint8)
            im = Image.fromarray(img_arr)
            im.save(os.path.join(save_root, 'render','%05d.png' % i))

    print('Finish!')
