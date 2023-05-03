# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, y_pred, real_image_flag):
        if real_image_flag:
            target = torch.ones_like(y_pred)
        else:
            target = torch.zeros_like(y_pred)
        return self.loss(y_pred, target)
