# extract perceptual features from the pre-trained Vgg16 network
# these features are used for the perceptual loss function (https://arxiv.org/abs/1603.08155)
#
# based on the code snippet of W. Falcon:
# https://gist.github.com/williamFalcon/1ee773c159ff5d76d47518653369d890

import torch
import torch.nn as nn
from torchvision import models

class Vgg16Features(nn.Module):

    def __init__(self,
                 requires_grad=False,
                 layers_weights=None):
        super(Vgg16Features, self).__init__()
        if layers_weights is None:
            self.layers_weights = [1.0, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        else:
            self.layers_weights = layers_weights

        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):

        h_0 = x.flatten(start_dim=1)
        h = self.slice1(x)
        h_relu1_2 = h.flatten(start_dim=1)
        h = self.slice2(h)
        h_relu2_2 = h.flatten(start_dim=1)
        h = self.slice3(h)
        h_relu3_3 = h.flatten(start_dim=1)
        h = self.slice4(h)
        h_relu4_3 = h.flatten(start_dim=1)

        h = torch.cat([self.layers_weights[0] * h_0,
                       self.layers_weights[1] * h_relu1_2,
                       self.layers_weights[2] * h_relu2_2,
                       self.layers_weights[3] * h_relu3_3,
                       self.layers_weights[4] * h_relu4_3], 1)

        return h

class PerceptualLossVGG(nn.Module):

    def __init__(self, layers_weights=None):
        super().__init__()
        self.vgg_features = Vgg16Features(layers_weights=layers_weights)
        self.loss = nn.L1Loss()

    def forward(self, y_pred, y_gt):
        y_pred_vgg_features = self.vgg_features(y_pred)
        y_gt_vgg_features = self.vgg_features(y_gt).detach()
        return self.loss(y_pred_vgg_features, y_gt_vgg_features)