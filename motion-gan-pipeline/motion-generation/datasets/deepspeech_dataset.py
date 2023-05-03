# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import sys
from unittest import skip

sys.path.append("..")

from datasets.base_dataset import BaseDataset
import scipy.io as sio
import torch
import librosa
import bisect
import os
import numpy as np

from funcs import utils


class DeepspeechDataset(BaseDataset):
    """ DECA datasets. currently, return 2D info and 3D tracking info.

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
        self.state = opt.dataset_type
        self.dataset_name = opt.dataset_names
        self.target_length = opt.time_frame_length
        self.sample_rate = opt.sample_rate
        self.fps = opt.FPS

        self.audioRF_history = opt.audioRF_history
        self.audioRF_future = opt.audioRF_future

        #Audio2Headpose flags
        self.A2H_receptive_field = opt.A2H_receptive_field
        self.A2H_item_length = self.A2H_receptive_field + self.target_length - 1 #204
        self.frame_future = opt.frame_future #15
        self.predict_length = opt.predict_length
        self.predict_len = int((self.predict_length - 1) / 2) #0

        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.total_len = 0
        start_point = self.A2H_receptive_field
        if opt.train_test_split:
            self.dataset_root = os.path.join(self.root, self.dataset_name, self.state)
            self.clip_names = sorted(os.listdir(self.dataset_root))
        
        else:
            self.dataset_root = self.root
            self.clip_names = [self.dataset_name]


        # check clips
        self.valid_clips = []
        for i in range(len(self.clip_names)):
            # check lenght of video
            name = self.clip_names[i]
            clip_root = os.path.join(self.dataset_root, name)
            n_frames = len(os.listdir(os.path.join(clip_root, 'frames')))

            if n_frames >= start_point + self.target_length: # added 25 because later they remove 25 and without it, crashes
                
                deepspeech_feature_path = os.path.join(clip_root, 'audio_feature')
        
                audio_features_len = len(os.listdir(deepspeech_feature_path))

                if audio_features_len > self.A2H_item_length:
                    self.valid_clips.append(i)
                
                else:
                    print(f'Audio {name} is too short and will not be used for training.')
            
            else:
                print(f'Clip {name} is too short and will not be used for training.')
        
        self.clip_nums = len(self.valid_clips)
        print(f'Total clips for training: {self.clip_nums}')
        
        # main info
        self.audio = [''] * self.clip_nums
        self.audio_features = [''] * self.clip_nums
        self.feats = [''] * self.clip_nums
        self.exps = [''] * self.clip_nums
        self.pts3d = [''] * self.clip_nums
        self.rot_angles = [''] * self.clip_nums
        self.trans = [''] * self.clip_nums
        self.headposes = [''] * self.clip_nums
        self.velocity_pose = [''] * self.clip_nums
        self.acceleration_pose = [''] * self.clip_nums
        self.mean_trans = [''] * self.clip_nums
        if self.state == 'Test':
            self.landmarks = [''] * self.clip_nums
        # meta info
        self.start_point = [''] * self.clip_nums
        self.end_point = [''] * self.clip_nums
        self.len = [''] * self.clip_nums
        self.sample_start = []
        self.clip_valid = ['True'] * self.clip_nums
        self.invalid_clip = []

        self.mouth_related_indices = np.concatenate([np.arange(4, 11), np.arange(46, 64)])

        # if opt.use_delta_pts:
        #     self.pts3d_mean = np.load(os.path.join(self.dataset_root, 'mean_pts3d.npy'))

        for i in range(len(self.valid_clips)):
            name = self.clip_names[self.valid_clips[i]]
            clip_root = os.path.join(self.dataset_root, name)
            deepspeech_feature_path = os.path.join(clip_root, 'audio_feature')

            # load deepspeech
            self.audio_features[i] = torch.from_numpy(np.stack([np.load(os.path.join(deepspeech_feature_path, ds)).flatten() for ds in sorted(os.listdir(deepspeech_feature_path))], axis=0))
            
            # 3D landmarks & headposes
            self.start_point[i] = start_point # They had 300 at 60 fps
            fit_data_path = os.path.join(clip_root, 'track_params.pt')
            fit_data = torch.load(fit_data_path)
            self.fit_data = fit_data

            self.rot_angles[i] = fit_data['euler']
            
            # # change -180~180 to 0~360
            # rot_change = self.rot_angles[i][:, 0] < 0
            # self.rot_angles[i][rot_change, 0] += 360
            # self.rot_angles[i][:, 0] -= 180  # change x axis direction

            # use delta translation
            self.mean_trans[i] = fit_data['trans'].mean(axis=0)
            self.trans[i] = fit_data['trans'] - self.mean_trans[i]

            self.headposes[i] = np.concatenate([self.rot_angles[i], self.trans[i]], axis=1)
            self.velocity_pose[i] = np.concatenate([np.zeros(6)[None, :], self.headposes[i][1:] - self.headposes[i][:-1]])
            self.acceleration_pose[i] = np.concatenate([np.zeros(6)[None, :], self.velocity_pose[i][1:] - self.velocity_pose[i][:-1]])

            # print(clip_root)
            # print('Headposes: ', self.headposes[i].shape)
            # print('Audio: ', self.audio_features[i].size())

            total_frames = min(self.trans[i].shape[0] - 25, self.audio_features[i].shape[0] - 25) # They had -60 (1 second at 60 fps) # Crashes without 25
            # print(f'Total Frames for clip {i}: {total_frames}')

            valid_frames = total_frames - self.start_point[i] 
            self.len[i] = valid_frames - self.target_length # ??? They had - 400 (6,66 s at 60 fps)
            if i == 0:
                self.sample_start.append(0)
            else:
                prev_start = self.sample_start[-1]
                len_prev = self.len[i - 1]
                self.sample_start.append(prev_start + len_prev - 1)
            self.total_len += np.int32(np.floor(self.len[i]))

   
    def __getitem__(self, index):
        # recover real index from compressed one
        index_real = np.int32(index)
        # find which audio file and the start frame index
        file_index = bisect.bisect_right(self.sample_start, index_real) -1
        current_frame = index_real - self.sample_start[file_index] + self.start_point[file_index]
        current_target_length = self.target_length

        # find the history info start points
        A2H_history_start = current_frame - self.A2H_receptive_field
        A2H_item_length = self.A2H_item_length # 204
        A2H_receptive_field = self.A2H_receptive_field

        
        A2Hsamples = np.zeros([A2H_item_length, 16 * 29]) #deepspeech
        for i in range(A2H_item_length - 1):
            try:
                A2Hsamples[i] = self.audio_features[file_index][A2H_history_start + i]
            
            except IndexError:
                print('### ERROR AUDIO ###')
                print('Name: ', self.clip_names[self.valid_clips[file_index]])
                print('Current Frame: ', current_frame)
                print('Data Index: ', index_real)
                print('Starts at: ', self.sample_start[file_index])
                print('Len: ', self.len[file_index])
                print(A2H_history_start + i)
                print(self.audio_features[file_index].size())
                print(self.headposes[file_index].shape)
                exit()

        if self.predict_len == 0:
            try:
                target_headpose = self.headposes[file_index][
                                    A2H_history_start + A2H_receptive_field: A2H_history_start + A2H_item_length + 1] # [current frame: current frame + target]
                history_headpose = self.headposes[file_index][A2H_history_start: A2H_history_start + A2H_item_length]
                history_headpose = history_headpose.reshape(A2H_item_length, -1) # [current_frame - self.A2H_receptive_field : current frame + target -1]

                target_velocity = self.velocity_pose[file_index][
                                    A2H_history_start + A2H_receptive_field: A2H_history_start + A2H_item_length + 1]
                history_velocity = self.velocity_pose[file_index][
                                    A2H_history_start: A2H_history_start + A2H_item_length].reshape(A2H_item_length,
                                                                                                    -1)
                target_info = torch.from_numpy(
                    np.concatenate([target_headpose, target_velocity], axis=1).reshape(current_target_length,
                                                                                        -1)).float()
            except ValueError:
                print('### ERROR HEAD ###')
                print('Name: ', self.clip_names[self.valid_clips[file_index]])
                print('Current Frame: ', current_frame)
                print('Data Index: ', index_real)
                print('Starts at: ', self.sample_start[file_index])
                print('Len: ', self.len[file_index])
                print(target_headpose.size)
                exit()

        else:
            history_headpose = self.headposes[file_index][
                                A2H_history_start: A2H_history_start + A2H_item_length]
            history_headpose = history_headpose.reshape(A2H_item_length, -1) #Same

            history_velocity = self.velocity_pose[file_index][
                                A2H_history_start: A2H_history_start + A2H_item_length]
            history_velocity = history_velocity.reshape(A2H_item_length, -1) #Same

            target_headpose_ = self.headposes[file_index][
                                A2H_history_start + A2H_receptive_field - self.predict_len: A2H_history_start + A2H_item_length + 1 + self.predict_len + 1] # [current frame - predict_len: current frame + target + predict_len + 1]
            target_headpose = np.zeros([current_target_length, self.predict_length, target_headpose_.shape[1]])
            for i in range(current_target_length):
                target_headpose[i] = target_headpose_[i: i + self.predict_length]
            target_headpose = target_headpose  # .reshape(current_target_length, -1, order='F')

            target_velocity_ = self.headposes[file_index][
                                A2H_history_start + A2H_receptive_field - self.predict_len: A2H_history_start + A2H_item_length + 1 + self.predict_len + 1]
            target_velocity = np.zeros([current_target_length, self.predict_length, target_velocity_.shape[1]])
            for i in range(current_target_length):
                target_velocity[i] = target_velocity_[i: i + self.predict_length]
            target_velocity = target_velocity  # .reshape(current_target_length, -1, order='F')

            target_info = torch.from_numpy(
                np.concatenate([target_headpose, target_velocity], axis=2).reshape(current_target_length,
                                                                                    -1)).float()

        A2Hsamples = torch.from_numpy(A2Hsamples).float()

        history_info = torch.from_numpy(np.concatenate([history_headpose, history_velocity], axis=1)).float()

        # [item_length, mel_channels, mel_width], or [item_length, APC_hidden_size]

        return A2Hsamples, history_info, target_info


    def __len__(self):
        return self.total_len
        # return self.total_len - 2000  # Hardcoded to make it work... does not work with short videos



