# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
from os.path import join
from turtle import color
import yaml
import argparse
import numpy as np
from options.test_audio2headpose_options import TestOptions
from datasets import create_dataset
from util.cfgnode import CfgNode
from torch.functional import F

from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
from scipy.stats import pearsonr


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
    target_dir= './visuals/Correlation/'
    os.makedirs(target_dir, exist_ok=True)

    total_movement = []
    total_emotions = []
    total_valence = []
    total_arousal = []

    emotion_labels = [
        'Neutral',
        'Happy',
        'Sad',
        'Surprise',
        'Fear',
        'Disgust',
        'Anger',
        'Contempt',
    ]

    for file_index in tqdm(range(len(dataset))):

        test_name = dataset.clip_names[file_index]

        name, flame, frames, audio_packed, audio_features, emotions, landmarks, exps, headposes, velocity_pose, fit_data_path = dataset[file_index]

        try:
            # Compute avarage totale movement (sum of all movement divided by number of frames)
            total_i = np.sum(np.abs(velocity_pose), axis=0) / headposes.shape[0]
            total_movement.append(total_i)

            avg_valence = np.mean(np.stack([frame.item()['valence'].flatten() for frame in emotions], axis=0))
            avg_arousal = np.mean(np.stack([frame.item()['arousal'].flatten() for frame in emotions], axis=0))
            avg_class = np.mean([frame.item()['expr_classification'] for frame in emotions], axis=0)
        
        except:
            print(f'Error in: {test_name}')
            pass
        
        avg_class = torch.tensor(avg_class)
        softmax = F.softmax(avg_class)
        top_expr = torch.argmax(softmax, dim=1)

        total_emotions.append(top_expr.numpy()[0])
        total_valence.append(avg_valence)
        total_arousal.append(avg_arousal)

    total_emotions = np.array(total_emotions)
    total_movement = np.array(total_movement)
    total_arousal = np.array(total_arousal)
    total_valence = np.array(total_valence)

    # Emotion distribution
    plt.title(f'Emotion Distribution out of 100 videos')
    xticks = np.arange(8) + 0.5
    plt.xticks(xticks, emotion_labels, rotation=0)  # Set text labels and properties.
    plt.yticks([])
    # Make histogram
    n, bins, patches = plt.hist(total_emotions, bins=np.arange(8), label=emotion_labels)
    # We'll color code by height, but you could use any scalar
    fracs = n / n.max()
    # we need to normalize the data to 0..1 for the full range of the colormap
    norm = colors.Normalize(fracs.min(), fracs.max())

    for idx, value in enumerate(n):
        if value > 0:
            plt.text(xticks[idx], value+1, int(value), ha='center')

    for thisfrac, thispatch in zip(fracs, patches):
        color = plt.cm.viridis(norm(thisfrac))
        thispatch.set_facecolor(color)

    plt.savefig(f'{target_dir}Emotion_distr.png') 
    plt.close()
    
    for idx, angle in enumerate(['Pitch', 'Roll', 'Yaw']):
        
        # Emotion
        plt.xticks(np.arange(8), emotion_labels, rotation=0)  # Set text labels and properties.
        plt.title(f'{angle} Angle')
        plt.scatter(x=total_emotions, y=total_movement[:, idx])  
        plt.ylabel('Avg total movement')
        plt.savefig(f'{target_dir}{angle}_Emotions.png') 
        plt.close()

        # Arousal
        r, p = pearsonr(x=total_arousal, y=total_movement[:, idx])
        r = round(r, 3)
        p = round(p, 3)
        x = np.linspace(np.min(total_arousal), np.max(total_arousal), 1000)

        plt.title(f'{angle} Angle and Arousal\n Correlation: {int(r*100)}%, p-value: {int(p*100)}%')
        plt.scatter(x=total_arousal, y=total_movement[:, idx])  
        plt.plot(x, (x * r) + np.mean(total_movement[:, idx]), linestyle='solid', color='red')
        plt.xlabel('Arousal')
        plt.ylabel('Avg total movement')
        plt.savefig(f'{target_dir}{angle}_Arousal.png') 
        plt.close()
        
        # Valence
        r, p = pearsonr(x=total_valence, y=total_movement[:, idx])
        r = round(r, 3)
        p = round(p, 3)
        x = np.linspace(np.min(total_valence), np.max(total_valence), 1000)

        plt.title(f'{angle} Angle and Valence\n Correlation: {int(r*100)}%, p-value: {int(p*100)}%')
        plt.scatter(x=total_valence, y=total_movement[:, idx])  
        plt.plot(x, (x * r) + np.mean(total_movement[:, idx]), linestyle='solid', color='red')
        plt.xlabel('Valence')
        plt.ylabel('Avg total movement')
        plt.savefig(f'{target_dir}{angle}_Valence.png') 
        plt.close()

    print('Finish!')
