# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import sys

sys.path.append("..")

from datasets.base_dataset import BaseDataset
import torch
import librosa
import os
import numpy as np


def load_audio_expressions(clip_root):
    audio_expr_path = os.path.join(clip_root, 'audio_expr')

    audio_expr = []
    for i in range(len(os.listdir(audio_expr_path))):
        audio_expr.append(np.load( os.path.join(audio_expr_path, f'audio_expression_{i}.npy'))[0])
    try:
        audio_expr = np.stack(audio_expr, axis=0)
        return torch.from_numpy(audio_expr)
        
    except ValueError:
        return None


class AudioDataset(BaseDataset):
    """ 

        # for wavenet:
        #           |----receptive_field----|
        #                                 |--output_length--|
        # example:  | | | | | | | | | | | | | | | | | | | | |
        # target:                           | | | | | | | | | |

    """

    def __init__(self, opt):
        # save the option and datasets root
        BaseDataset.__init__(self, opt)
        self.isTrain = self.opt.isTrain
        self.state = opt.phase
        self.dataset_name = opt.dataset_names
        self.target_length = opt.time_frame_length
        self.sample_rate = opt.sample_rate
        self.fps = opt.FPS

        self.audioRF_history = opt.audioRF_history
        self.audioRF_future = opt.audioRF_future
        # self.compute_mel_online = opt.compute_mel_online
        # self.feature_name = opt.feature_name

        self.audio_samples_one_frame = self.sample_rate / self.fps
        self.frame_jump_stride = opt.frame_jump_stride
        self.augment = False
        self.task = opt.task
        self.item_length_audio = int((self.audioRF_history + self.audioRF_future) / self.fps * self.sample_rate)

        #Audio2Headpose flags
        self.A2H_receptive_field = opt.A2H_receptive_field
        self.A2H_item_length = self.A2H_receptive_field + self.target_length - 1 #204
        self.audio_window = opt.audio_windows
        self.half_audio_win = int(self.audio_window / 2) #1
        self.frame_future = opt.frame_future #15
        self.predict_length = opt.predict_length
        self.predict_len = int((self.predict_length - 1) / 2) #0

        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.total_len = 0

        if opt.train_test_split:
            self.dataset_root = os.path.join(self.root, self.dataset_name, self.state)
            self.clip_names = sorted(os.listdir(self.dataset_root))
        
        else:
            self.dataset_root = self.root
            self.clip_names = [self.dataset_name]

            print(self.dataset_root)
            print(self.clip_names)

        # check clips
        self.valid_clips = []
        for i in range(len(self.clip_names)):
            # check lenght of video
            name = self.clip_names[i]
            clip_root = os.path.join(self.dataset_root, name)

            deepspeech_feature_path = os.path.join(clip_root, 'audio_feature')
    
            audio_features_len = len(os.listdir(deepspeech_feature_path))

            # if audio_features_len > self.A2H_item_length:
            #     self.valid_clips.append(i)
            
            # else:
            #     print(f'Audio {name} is too short and will not be used for training.')
            
            self.valid_clips.append(i)
            
        
        self.clip_nums = len(self.valid_clips)
        print(f'Total clips for training: {self.clip_nums}')
        
        # main info
        self.audio = [''] * self.clip_nums
        self.audio_features = [''] * self.clip_nums
        self.fit_data = [''] * self.clip_nums

        # audio expressions
        self.audio_expr = [''] * self.clip_nums


        for i in range(len(self.valid_clips)):
            name = self.clip_names[i]
            clip_root = os.path.join(self.dataset_root, name)
            deepspeech_feature_path = os.path.join(clip_root, 'audio_feature')

            # Load audio
            try:
                audio_path = os.path.join(clip_root, name + '.wav')
                self.audio[i], _ = librosa.load(audio_path, sr=self.sample_rate)
            
            except FileNotFoundError:
                pass

            # load deepspeech
            self.audio_features[i] = torch.from_numpy(np.stack([np.load(os.path.join(deepspeech_feature_path, ds)).flatten() for ds in os.listdir(deepspeech_feature_path)], axis=0))

            
            try:
                fit_data_path = os.path.join(clip_root, 'track_params.pt')
                fit_data = torch.load(fit_data_path)
                self.fit_data[i] = fit_data
                # load audio expr
                self.audio_expr[i] = load_audio_expressions(clip_root)

            except FileNotFoundError:
                pass

    def __getitem__(self, index):
        return index
    
    def __len__(self):
        return len(self.audio_features)

