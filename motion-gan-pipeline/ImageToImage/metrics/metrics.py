# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import torch
import torch.nn as nn
from ignite.metrics import MeanAbsoluteError, MeanSquaredError, SSIM, PSNR, FID

from pytorch_fid.inception import InceptionV3

class Metrics():
    def __init__(self, args, device):
        
        # FID score model
        dims = 2048
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
        model = InceptionV3([block_idx]).to(device)
        wrapper_model = WrapperInceptionV3(model)
        wrapper_model.eval()

        available_metrics = {
            'mean_absolute_error': lambda: MeanAbsoluteError(device=device),
            'mean_squared_error': lambda: MeanSquaredError(device=device),
            'structual_similarity_index_measure': lambda: SSIM(data_range=1.0, device=device),
            'peak_signal_to_noise_ratio': lambda: PSNR(data_range=1.0, device=device),
            'frechet_inception_distance': lambda: FID(num_features=dims, feature_extractor=wrapper_model, device=device)
        }
        self.metric_names = args.metric_names
        self.metric_weights = args.metric_weights
        self.metric_list = []
        for metric_name in self.metric_names:
            if metric_name in available_metrics:
                self.metric_list.append(available_metrics[metric_name]())

    def update(self, y_pred, y_gt):
        for metric_fn in self.metric_list:
            metric_fn.update((y_pred, y_gt))

    def reset(self):
        for metric_fn in self.metric_list:
            metric_fn.reset()

    def compute(self):
        results = {}
        metric_combined = 0
        for idx, (metric_fn, metric_name) in enumerate(zip(self.metric_list, self.metric_names)):
            if metric_name == 'peak_signal_to_noise_ratio':
                results[metric_name] = metric_fn.compute().item()    
            else:
                results[metric_name] = metric_fn.compute()
            metric_combined += self.metric_weights[idx]*results[metric_name]
        return results, metric_combined


# wrapper class as feature_extractor
class WrapperInceptionV3(nn.Module):

    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y
