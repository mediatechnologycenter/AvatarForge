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
sys.path.append('../preprocessing')
from preprocessing.face_tracking.FLAME.FLAME import FLAME
from preprocessing.face_tracking.FLAME.config import cfg as FLAME_cfg
from preprocessing.face_tracking.FLAME.lbs import vertices2landmarks
from preprocessing.face_tracking.render_3dmm import Render_FLAME
from preprocessing.face_tracking.util import *


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


def load_audio_expressions(new_opt):

    clip_names = os.listdir(self.dataset_root)

    # check clips
    self.valid_clips = []
    for i in range(len(self.clip_names)):
        # check lenght of video
        name = self.clip_names[i]
        clip_root = os.path.join(self.dataset_root, name)
        n_frames = len(os.listdir(os.path.join(clip_root, 'frames')))
        if n_frames >= start_point + self.target_length + 25: # added 25 because later they remove 25 and without it, crashes
            audio_path = os.path.join(clip_root, name + '.wav')
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)

            if len(audio) >= self.frame_future + self.A2H_item_length:
                self.valid_clips.append(i)
            
            else:
                print(f'Audio {name} is too short and will not be used for training.')
        
        else:
            print(f'Clip {name} is too short and will not be used for training.')
    
    self.clip_nums = len(self.valid_clips)
    print(f'Total clips for training: {self.clip_nums}')

    return

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
    opt.dataset_mode = 'audio'  # for testing
    opt.dataroot = cfg.experiment.dataroot
    opt.dataset_names = cfg.experiment.dataset_names
    opt.FPS = cfg.experiment.fps
    opt.serial_batches = True

    # save to the disk
    Test_parser.print_options(opt)

    # Set device
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if len(
            opt.gpu_ids) > 0 else torch.device('cpu')

    # Load data
    # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset(opt)
    try:
        fit_data = dataset.dataset.fit_data

    except AttributeError:
        focal = torch.from_numpy([900.]).to(device).double()
        expr_para = torch.zeros(size=(1,53)).to(device).double()
        id_para = torch.zeros(size=(1,100)).to(device).double()

    # Load Model
    print('---------- Loading Model: {} -------------'.format(opt.task))
    Audio2Headpose = create_model(opt)
    Audio2Headpose.setup(opt)
    Audio2Headpose.eval()

    stop_at = 10

    for iter, file_index in enumerate(dataset.dataset.valid_clips):

        if iter >= stop_at:
            break

        test_name = dataset.dataset.clip_names[dataset.dataset.valid_clips[file_index]]
        #____________________________________________________#
        print('Generating movement for video: ', test_name)
        audio = dataset.dataset.audio[file_index]
        audio_features = dataset.dataset.audio_features[file_index]
        try:
            audio_expr = dataset.dataset.audio_expr[file_index]
        except argparse.ArgumentError:
            pass

        # Audio2Headpose
        print('Headpose inference...')
        # set history headposes as zero
        pre_headpose = np.zeros(opt.A2H_wavenet_input_channels, np.float32)
        pred_Head = Audio2Headpose.generate_sequences(
            audio_features, pre_headpose, fill_zero=True, sigma_scale=0.3, opt=opt)

        # Build FLAME and Renderer
        try:
            h = fit_data[iter]['h']
            w = fit_data[iter]['w']
            focal = fit_data[iter]['focal']
            id_para = fit_data[iter]['id']
            # expr_para = fit_data[iter]['exp']
            expr_para = audio_expr
            nframe = min(expr_para.size()[0], pred_Head.shape[0])

        except KeyError:
            h = 720
            w = 1280
            nframe = pred_Head.shape[0]
            expr_para = expr_para.repeat(nframe)

        cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).cpu()

        model_3dmm = FLAME(FLAME_cfg.model)
        renderer = Render_FLAME(model_3dmm.faces_tensor, focal, h, w, 1, device)

        # Postprocessing
        if fit_data:
            
            og_rot = fit_data[iter]['euler'].numpy()
            # og_rot[:, 0] += 180
            og_trans = fit_data[iter]['trans'].numpy()

            mean_translation = og_trans.mean(axis=0)

            # headpose back to og
            Head_smooth_sigma = [5, 10]

            pred_headpose = utils.headpose_smooth(
                pred_Head[:, :6], Head_smooth_sigma).astype(np.float32)

            # pred_headpose = pred_Head[:, :6].astype(np.float32)
            pred_headpose[:, 3:] += mean_translation
            # pred_headpose[:, 0] += 180

        else:
            pred_headpose = pred_Head[:, :6].astype(np.float32)

        # Make images
        save_root = os.path.join('./results/', opt.name, test_name)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
            os.makedirs(os.path.join(save_root, 'landmarks'))
        
        np.save(os.path.join(save_root, 'headposes.npy'), pred_headpose)

        for i in tqdm(range(nframe), desc='rendering: '):
            R = torch.from_numpy(pred_headpose[i, 0:3]).unsqueeze(
                0).to(device).double()
            t = torch.from_numpy(pred_headpose[i, 3:6]).unsqueeze(
                0).to(device).double()
            id = id_para.to(device).double()
            expr = expr_para[i].unsqueeze(0).to(device).double()
            

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

            rott_geo = model_3dmm.forward_geo(id, expr, R, t)
            landmarks3d = model_3dmm.get_3dlandmarks(
                id, expr, R, t, focal, cxy).cpu()
            proj_geo = proj_pts(landmarks3d, focal, cxy)
            np.save(os.path.join(save_root, 'landmarks', f'ldk_{i}.npy'), proj_geo[0].detach().cpu().numpy())

            # # Porj points
            # colormap_red = plt.cm.Reds
            # for num, lin in enumerate(np.linspace(0, 0.9, len(proj_geo[0, :, 0]))):
            #     plt.scatter(proj_geo[0, num, 0].detach().cpu(),
            #                 proj_geo[0, num, 1].detach().cpu(),
            #                 color=colormap_red(lin),
            #                 s=10)

            # plt.show()

            sel_pts3D = vertices2landmarks(rott_geo,
                                        model_3dmm.faces_tensor,
                                        model_3dmm.full_lmk_faces_idx.repeat(
                                            1, 1),
                                        model_3dmm.full_lmk_bary_coords.repeat(1, 1, 1))

            render_imgs = renderer(rott_geo.float(), model_3dmm.faces_tensor)
            img_arr = render_imgs[0, :, :, :3].cpu().numpy()
            # img_arr = img_arr / 255.
            # img_arr = img_arr.astype(np.uint8)
            # cv2.imwrite(os.path.join(save_root, f'flame_{i}.jpg'), img_arr)

            # # plt.imshow(img_arr)
            # # plt.show()
            from PIL import Image
            img_arr *= 255
            img_arr = img_arr.astype(np.uint8)
            im = Image.fromarray(img_arr)
            im.save(os.path.join(save_root, f'flame_{i}.jpg'))

        # make videos
        # generate corresponding audio, reused for all results
        tmp_audio_path = os.path.join(save_root, 'tmp.wav')
        tmp_audio_clip = audio[: np.int32(nframe * opt.sample_rate / opt.FPS)]
        sf.write(tmp_audio_path, tmp_audio_clip, opt.sample_rate)
        final_path = os.path.join(save_root, test_name + '.avi')
        write_video_with_audio(tmp_audio_path, final_path, 'flame_', h, w)

        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        _img_paths = list(map(lambda x:str(x), list(Path(save_root).glob('*.jpg'))))
        for i in tqdm(range(len(_img_paths)), desc='deleting intermediate images'):
            os.remove(_img_paths[i])

    print('Finish!')
