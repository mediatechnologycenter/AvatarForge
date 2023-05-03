# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import torch
import torch.nn as nn
from .perceptual_loss_vgg import PerceptualLossVGG
from .gan_loss import GANLoss

class Loss(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        available_losses = {
            'perceptual_loss_vgg': lambda: PerceptualLossVGG(layers_weights=[1.0/32,1.0/16,1.0/8,1.0/4,1.0]), #[1.0/32,1.0/16,1.0/8,1.0/4,1.0]
            'l1_perceptual_loss': lambda: nn.L1Loss()
        }
        self.loss_names = args.loss_names
        self.loss_weights = args.loss_weights
        self.loss_list = []
        assert(len(self.loss_names)==len(self.loss_weights))
        for loss_name in self.loss_names:
            if loss_name in available_losses:
                self.loss_list.append(available_losses[loss_name]().to(device))
    
    def eval(self):
        for loss in self.loss_list:
            loss.eval()

    def forward(self, y_pred, y_gt):
        loss = 0
        for k, loss_fn in enumerate(self.loss_list):
            loss += self.loss_weights[k]*loss_fn(y_pred, y_gt)
        return loss
