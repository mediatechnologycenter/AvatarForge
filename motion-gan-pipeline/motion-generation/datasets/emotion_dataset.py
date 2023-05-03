# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from email.mime import audio
import sys

sys.path.append("..")

from datasets.base_dataset import BaseDataset
import scipy.io as sio
import torch
import librosa
import os
import numpy as np

from funcs import utils


class EmotionDataset(BaseDataset):
    """ 
    Load Emotions and headposes.
    """

    def __init__(self, opt):
        # save the option and datasets root
        BaseDataset.__init__(self, opt)
        self.isTrain = self.opt.isTrain
        self.state = opt.dataset_type
        self.dataset_name = opt.dataset_names
        self.fps = opt.FPS

        self.dataset_root = os.path.join(self.root, self.dataset_name, self.state)
        self.clip_names = sorted(os.listdir(self.dataset_root))

        self.clip_nums = len(self.clip_names)
        print(f'Total clips for training: {self.clip_nums}')
        
        # main info
        self.frames = [''] * self.clip_nums
        self.flame = [''] * self.clip_nums
        self.audio = [''] * self.clip_nums
        self.audio_features = [''] * self.clip_nums
        self.emotions = [''] * self.clip_nums
        self.exps = [''] * self.clip_nums
        self.pts3d = [''] * self.clip_nums
        self.rot_angles = [''] * self.clip_nums
        self.trans = [''] * self.clip_nums
        self.headposes = [''] * self.clip_nums
        self.velocity_pose = [''] * self.clip_nums
        self.acceleration_pose = [''] * self.clip_nums
        self.mean_trans = [''] * self.clip_nums
        self.landmarks = [''] * self.clip_nums
        self.fit_data_path = [''] * self.clip_nums
        
        self.len = [''] * self.clip_nums
        self.clip_valid = ['True'] * self.clip_nums


        for i in range(len(self.clip_names)):
            name = self.clip_names[i]
            clip_root = os.path.join(self.dataset_root, name)

            # Paths to subfolders
            frames_path = os.path.join(clip_root, 'frames')
            flame_path = os.path.join(clip_root, 'debug/debug_render')
            deepspeech_feature_path = os.path.join(clip_root, 'audio_feature')
            emotions_path = os.path.join(clip_root, 'emotions')
            expr_path = os.path.join(clip_root, 'deca_expr')
            landmarks_path = os.path.join(clip_root, 'landmarks')
            fit_data_path = os.path.join(clip_root, 'track_params.pt')

            # load wav audio signal
            # Open wav file and read frames as bytes
            x, sr = librosa.load(os.path.join(clip_root, f'{name}.wav'))
            self.audio[i] = (x, sr)

            # load frames
            self.frames[i] = [os.path.join(frames_path, f'{f}.jpg') for f in range(len(os.listdir(frames_path)))]

            # load debug
            self.flame[i] = [os.path.join(flame_path, f'{f}.jpg') for f in range(len(os.listdir(flame_path)))]

            # load deepspeech
            self.audio_features[i] = torch.from_numpy(np.stack([np.load(os.path.join(deepspeech_feature_path, f'{ds}.deepspeech.npy')).flatten() for ds in range(len(os.listdir(deepspeech_feature_path)))], axis=0))
            
            # load emotions
            try:
                self.emotions[i] = [np.load(os.path.join(emotions_path, em), allow_pickle=True) for em in sorted(os.listdir(emotions_path))]
            
            except FileNotFoundError:
                self.emotions[i] = None
            
            # load landmarks
            self.landmarks[i] = torch.from_numpy(np.stack([np.loadtxt(os.path.join(landmarks_path, ldk)) for ldk in sorted(os.listdir(landmarks_path))], axis=0))

            # load expressions
            self.exps[i] = torch.from_numpy(np.stack([np.load(os.path.join(expr_path, f'{exp}.npy')) for exp in range(len(os.listdir(expr_path)))], axis=0))

            # 3D landmarks & headposes
            fit_data = torch.load(fit_data_path)
            self.fit_data = fit_data
            self.fit_data_path[i] = fit_data_path
            self.rot_angles[i] = fit_data['euler']
    
            self.mean_trans[i] = fit_data['trans'].mean(axis=0)
            self.trans[i] = fit_data['trans'] - self.mean_trans[i]
            self.headposes[i] = np.concatenate([self.rot_angles[i], self.trans[i]], axis=1)
            self.velocity_pose[i] = np.concatenate([np.zeros(6)[None, :], self.headposes[i][1:] - self.headposes[i][:-1]])
            self.acceleration_pose[i] = np.concatenate([np.zeros(6)[None, :], self.velocity_pose[i][1:] - self.velocity_pose[i][:-1]])


        return
   
    def __getitem__(self, index):

        data = (
            self.clip_names[index],
            self.flame[index],
            self.frames[index],
            self.audio[index],
            self.audio_features[index],
            self.emotions[index],
            self.landmarks[index],
            self.exps[index],
            self.headposes[index],
            self.velocity_pose[index],
            self.fit_data_path[index],
        )

        return data

    def __len__(self):
        return self.clip_nums



