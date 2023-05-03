"""This module contains functions for geometry transform and camera projection"""
import torch
import torch.nn as nn
import numpy as np
from pytorch3d.transforms import euler_angles_to_matrix

def euler2rot(euler_angle):
    rot = torch.deg2rad(euler_angle)
    rot = euler_angles_to_matrix(rot, convention='XYZ')
    return rot


def rot_trans_geo(geometry, rot, trans):
    rott_geo = torch.bmm(rot, geometry.permute(0, 2, 1)) + trans.view(-1, 3, 1)
    return rott_geo.permute(0, 2, 1)


def euler_trans_geo(geometry, euler, trans):
    rot = euler2rot(euler)
    return rot_trans_geo(geometry, rot, trans)


def proj_geo(rott_geo, camera_para):
    fx = camera_para[:, 0]
    fy = camera_para[:, 0]
    cx = camera_para[:, 1]
    cy = camera_para[:, 2]

    X = rott_geo[:, :, 0]
    Y = rott_geo[:, :, 1]
    Z = rott_geo[:, :, 2]

    fxX = fx[:, None]*X
    fyY = fy[:, None]*Y

    proj_x = -fxX/Z + cx[:, None]
    proj_y = fyY/Z + cy[:, None]

    return torch.cat((proj_x[:, :, None], proj_y[:, :, None], Z[:, :, None]), 2)
