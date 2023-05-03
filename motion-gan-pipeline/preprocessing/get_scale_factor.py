# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import numpy as np
import torch
import os
from preprocessing.face_tracking.FLAME.FLAME import FLAME, FLAMETex
from preprocessing.face_tracking.FLAME.config import cfg
from pytorch3d.io import IO, save_obj
import trimesh
from preprocessing.face_tracking.facemodel import Face_3DMM

# Create FLAME model
sel_num = 1
id_dim, exp_dim, tex_dim = 100, 50, 50

id_para = torch.zeros((1, id_dim), dtype=torch.float32).cuda()
id_para = id_para.expand(sel_num, -1)
exp_para = torch.zeros((sel_num, exp_dim), dtype=torch.float32).cuda()
euler_angle = torch.zeros((sel_num, 3), dtype=torch.float32).cuda()
trans = torch.zeros((sel_num, 3), dtype=torch.float32).cuda()
focal_length = torch.tensor([1100], dtype=torch.float32).cuda()
w, h = (512, 512)
cxy = torch.tensor((w/2.0, h/2.0), dtype=torch.float32).cuda()

# Make model
FLAME_3dmm = FLAME(cfg.model)

flame_landmarks = FLAME_3dmm.get_3dlandmarks(id_para, exp_para, euler_angle, trans, focal_length, cxy)
flame_geometry = FLAME_3dmm.forward_geo(id_para, exp_para, euler_angle, trans)

print(flame_geometry.size())
print(flame_landmarks.size())

# Save mesh
save_path= 'test_FLAME_mesh.obj'
verts = flame_geometry.cpu()[0]
faces = FLAME_3dmm.faces_tensor.cpu()
save_obj(save_path, verts=verts, faces=faces)

# Save landmarks
save_path = 'test_FLAME_landmarks.obj'
flame_landmarks = flame_landmarks.cpu()[0]
pcl = trimesh.PointCloud(flame_landmarks)
pcl.export(save_path)


############################

# Create weird Basel model

dir_path = os.path.dirname(os.path.realpath(__file__))
sel_num = 1
id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650

id_para = torch.zeros((1, id_dim), dtype=torch.float32).cuda()
id_para = id_para.expand(sel_num, -1)
exp_para = torch.zeros((sel_num, exp_dim), dtype=torch.float32).cuda()
euler_angle = torch.zeros((sel_num, 3), dtype=torch.float32).cuda()
trans = torch.zeros((sel_num, 3), dtype=torch.float32).cuda()
focal_length = torch.tensor([1100], dtype=torch.float32).cuda()
w, h = (512, 512)
cxy = torch.tensor((w/2.0, h/2.0), dtype=torch.float32).cuda()

Basel_3dmm = Face_3DMM(os.path.join(dir_path, 'face_tracking/3DMM'), id_dim, exp_dim, tex_dim, point_num)

basel_landmarks = Basel_3dmm.get_3dlandmarks(id_para, exp_para, euler_angle, trans, focal_length, cxy)
basel_geometry = Basel_3dmm.forward_geo(id_para, exp_para)

print(basel_geometry.size())
print(basel_landmarks.size())

# Save mesh
save_path= 'test_BASEL_mesh.obj'
verts = basel_geometry.cpu()[0]
topo_info = np.load(os.path.join(
            dir_path, 'face_tracking/3DMM', 'topology_info.npy'), allow_pickle=True).item()
faces = torch.as_tensor(topo_info['tris']).cuda().float()

save_obj(save_path, verts=verts, faces=faces)

# Save landmarks
save_path= 'test_BASEL_landmarks.obj'
basel_landmarks = basel_landmarks.cpu()[0]
pcl = trimesh.PointCloud(basel_landmarks)
pcl.export(save_path)

# find scale

def cal_lan_loss(proj_lan, gt_lan):
    return torch.mean((proj_lan-gt_lan)**2)

scale = torch.ones((1, 1), requires_grad=True)
optimizer = torch.optim.Adam([scale], lr=.1)

for i in range(200):
    scale_batch = scale.expand(68, 3)
    scaled_landmark_FLAME = torch.mul(flame_landmarks, scale_batch)
    loss_lan = cal_lan_loss(scaled_landmark_FLAME, basel_landmarks)

    loss = loss_lan
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i == 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.01


print('Final scale: ', scale)

# Save mesh
save_path= 'test_FLAME_scaled_mesh.obj'
scale = scale.expand(1, 3)
verts = torch.mul(flame_geometry[0].cpu(), scale)
faces = FLAME_3dmm.faces_tensor.cpu()
save_obj(save_path, verts=verts, faces=faces)
