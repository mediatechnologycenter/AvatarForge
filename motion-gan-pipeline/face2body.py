# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from genericpath import isfile
from math import floor
import os
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

class LandmarkDataset(Dataset):
    def __init__(self, landmarks_dir, body_dir, im_size, split):
        """
        Args:
            input_root_dir (string): Directory with all the input images.
            output_root_dir (string): Directory with all the output images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.landmarks_dir = landmarks_dir
        self.body_dir = body_dir
        self.w, self.h = im_size

        # set split for train test
        self.split_ratio = 0.8
        self.split = split

        self.total_num = len(os.listdir(self.landmarks_dir))

    def __len__(self):
        
        if self.split=='Train':
            return int(floor(self.total_num ) * self.split_ratio)
        
        else:
            return int(floor(self.total_num ) * (1 - self.split_ratio))

    def __getitem__(self, idx):

        if self.split=='Train':
            base = 0
        
        else:
            base = int(floor(self.total_num) * (self.split_ratio))

        idx += base

        landmark_path = os.path.join(self.landmarks_dir, '%05d.lms' % idx)
        body_path = os.path.join(self.body_dir, '%05d.npy' % idx)

        input_ldk = np.loadtxt(landmark_path)[:, :-1]
        output_ldk = np.load(body_path)[:, :-2]
        output_ldk = np.array([output_ldk[11], output_ldk[12], output_ldk[23], output_ldk[24]])

        # scale
        input_ldk[:, 0] = input_ldk[:, 0] / self.w
        input_ldk[:, 1] = input_ldk[:, 1] / self.h

        output_ldk[:, 0] = output_ldk[:, 0] / self.w
        output_ldk[:, 1] = output_ldk[:, 1] / self.h

        trans = transforms.ToTensor()
        sample = {
            'input_ldk': trans(input_ldk).flatten().float(), 
            'output_ldk': trans(output_ldk).flatten().float(), 
        }

        return sample


def train_regression_head2body(landmarks_dir, body_dir, checkpoint_path, im_size):

    train_dataset = LandmarkDataset(landmarks_dir, body_dir, im_size, "Train")
    train_dataloader = DataLoader(train_dataset, batch_size=train_dataset.__len__())

    test_dataset = LandmarkDataset(landmarks_dir, body_dir, im_size, "Test")
    test_dataloader = DataLoader(test_dataset, batch_size=test_dataset.__len__())

    # check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Running on", device)

    # Make linear model
    model = nn.Linear(77 * 2, 4 * 2).to(device) # 77 landmarks with 2 coordinates (x, y) -> 4 landmarks with 2 coordinates (x, y)    
    
    # Optimizer
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.1)

    # Loss Function
    loss_fn = F.mse_loss

    # Run optimisation
    for epoch in tqdm(range(25)):
        for batch in train_dataloader:
            # Generate predictions
            input_ldk = batch['input_ldk'].to(device)
            output_ldk = batch['output_ldk'].to(device)
            pred = model(input_ldk)

            # compute loss
            loss = loss_fn(pred, output_ldk)

            # Perform gradient descent
            loss.backward()
            opt.step()
            opt.zero_grad()

        lr_scheduler.step()

        if epoch % 10 == 0:
            with torch.no_grad():
                total_loss = 0.
                for batch in test_dataloader:
                    # Generate predictions
                    input_ldk = batch['input_ldk'].to(device)
                    output_ldk = batch['output_ldk'].to(device)
                    pred = model(input_ldk)

                    # compute loss
                    loss = loss_fn(pred, output_ldk)
                    total_loss += loss
                    
                    input_ldk = input_ldk.cpu().detach().reshape((-1, 77, 2)).numpy()
                    output_ldk = output_ldk.cpu().detach().reshape((-1, 4, 2)).numpy()
                    pred = pred.cpu().detach().reshape((-1, 4, 2)).numpy()
                    plt.scatter(input_ldk[0, :, 0], input_ldk[0, :, 1], marker='o', c='red')
                    plt.scatter(output_ldk[0, :, 0], output_ldk[0, :, 1], marker='o', c='green')
                    plt.scatter(pred[0, :, 0], pred[0, :, 1], marker='^', c='blue')

                    plt.savefig(os.path.splitext(checkpoint_path)[0] + '_test.png')
                    plt.close('all')

                print(f'Test Loss at epoch {epoch}: {total_loss.item()}')
        
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == '__main__':

    # load args 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_base', required=True)
    parser.add_argument('--target_name', required=True)
    parser.add_argument('--checkpoint_dir', required=True)

    inopt = parser.parse_args()

    landmarks_dir = os.path.join(inopt.dataset_base, 'debug/proj_landmarks')
    body_dir = os.path.join(inopt.dataset_base, 'body_pose')
    checkpoint_path = os.path.join(inopt.checkpoint_dir, inopt.target_name, 'head2body.pkl')


    if os.path.isfile(checkpoint_path):
        print('Head2body already trained!')
    
    else:
        tmp_img = Image.open(os.path.join(inopt.dataset_base, 'frames', '00000.jpg'))
        im_size = tmp_img.size
        train_regression_head2body(landmarks_dir, body_dir, checkpoint_path, im_size)

    
