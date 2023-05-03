# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn as nn
import numpy as np
import pickle
import torch.nn.functional as F

from .lbs import lbs, batch_rodrigues, vertices2landmarks, rot_mat_to_euler
from ..util import *


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, config):
        super(FLAME, self).__init__()
        print("creating the FLAME Decoder")
        with open(config.flame_model_path, 'rb') as f:
            ss = pickle.load(f, encoding='latin1')
            flame_model = Struct(**ss)

        self.scale_factor = torch.tensor(7.4254, dtype=torch.float).cuda()
        # self.scale_factor = torch.tensor(1.0, dtype=torch.float).cuda()

        # self.dtype = torch.float32
        self.dtype = torch.double

        self.register_buffer('faces_tensor', to_tensor(to_np(flame_model.f, dtype=np.int64), dtype=torch.long).cuda())
        # The vertices of the template model
        self.register_buffer('v_template', to_tensor(to_np(flame_model.v_template), dtype=self.dtype).cuda())
        # The shape components and expression
        shapedirs = to_tensor(to_np(flame_model.shapedirs), dtype=self.dtype)
        shapedirs = torch.cat([shapedirs[:,:,:config.n_shape], shapedirs[:,:,300:300+config.n_exp]], 2).cuda()
        self.register_buffer('shapedirs', shapedirs)
        # The pose components
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=self.dtype).cuda())
        # 
        self.register_buffer('J_regressor', to_tensor(to_np(flame_model.J_regressor), dtype=self.dtype).cuda())
        parents = to_tensor(to_np(flame_model.kintree_table[0])).long(); parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(to_np(flame_model.weights), dtype=self.dtype).cuda())

        # Fixing Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False).cuda()
        self.register_parameter('eye_pose', nn.Parameter(default_eyball_pose,
                                                         requires_grad=False))
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False).cuda()
        self.register_parameter('neck_pose', nn.Parameter(default_neck_pose,
                                                          requires_grad=False))

        default_pose = torch.zeros([1, config.n_pose], dtype=self.dtype, requires_grad=False).cuda()
        self.register_parameter('base_pose', nn.Parameter(default_pose,
                                                          requires_grad=False))

        # Static and Dynamic Landmark embeddings for FLAME
        lmk_embeddings = np.load(config.flame_lmk_embedding_path, allow_pickle=True, encoding='latin1')
        lmk_embeddings = lmk_embeddings[()]
        self.register_buffer('lmk_faces_idx', torch.from_numpy(lmk_embeddings['static_lmk_faces_idx']).long().cuda())
        self.register_buffer('lmk_bary_coords', torch.from_numpy(lmk_embeddings['static_lmk_bary_coords']).to(self.dtype).cuda())
        self.register_buffer('dynamic_lmk_faces_idx', lmk_embeddings['dynamic_lmk_faces_idx'].long())
        self.register_buffer('dynamic_lmk_bary_coords', lmk_embeddings['dynamic_lmk_bary_coords'].to(self.dtype))
        self.register_buffer('full_lmk_faces_idx', torch.from_numpy(lmk_embeddings['full_lmk_faces_idx']).long().cuda())
        self.register_buffer('full_lmk_bary_coords', torch.from_numpy(lmk_embeddings['full_lmk_bary_coords']).to(self.dtype).cuda())

        neck_kin_chain = []; NECK_IDX=1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
        
    def _find_dynamic_lmk_idx_and_bcoords(self, pose, dynamic_lmk_faces_idx,
                                          dynamic_lmk_b_coords,
                                          neck_kin_chain, dtype=torch.float32):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi,
                        max=39)).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    def _vertices2landmarks(self, vertices, faces, lmk_faces_idx, lmk_bary_coords):
        """
            Calculates landmarks by barycentric interpolation
            Input:
                vertices: torch.tensor NxVx3, dtype = torch.float32
                    The tensor of input vertices
                faces: torch.tensor (N*F)x3, dtype = torch.long
                    The faces of the mesh
                lmk_faces_idx: torch.tensor N X L, dtype = torch.long
                    The tensor with the indices of the faces used to calculate the
                    landmarks.
                lmk_bary_coords: torch.tensor N X L X 3, dtype = torch.float32
                    The tensor of barycentric coordinates that are used to interpolate
                    the landmarks

            Returns:
                landmarks: torch.tensor NxLx3, dtype = torch.float32
                    The coordinates of the landmarks for each mesh in the batch
        """
        # Extract the indices of the vertices for each face
        # NxLx3
        batch_size, num_verts = vertices.shape[:dd2]
        lmk_faces = torch.index_select(faces, 0, lmk_faces_idx.view(-1)).view(
            1, -1, 3).view(batch_size, lmk_faces_idx.shape[1], -1)

        lmk_faces += torch.arange(batch_size, dtype=torch.long).view(-1, 1, 1).to(
            device=vertices.device) * num_verts

        lmk_vertices = vertices.view(-1, 3)[lmk_faces]
        landmarks = torch.einsum('blfi,blf->bli', [lmk_vertices, lmk_bary_coords])
        return landmarks

    def seletec_3d68(self, vertices):
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                       self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
                                       self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1))
        return landmarks3d

    def get_3dlandmarks(self, shape_params, expression_params, euler_angle, trans, focal_length, cxy):

        vertices = self.forward_geo(shape_params, expression_params, euler_angle, trans, rot=True)

        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         self.full_lmk_faces_idx.repeat(bz, 1),
                                         self.full_lmk_bary_coords.repeat(bz, 1, 1))

        return landmarks3d.cuda()
    
    def get_3dlandmarks_special(self, shape_params, expression_params, euler_angle, trans, idx):

        vertices = self.forward_geo(shape_params, expression_params, euler_angle, trans, rot=True)

        # New vertices
        new_faces = np.array([idx])
        device = shape_params.get_device()
        new_faces_idx = torch.from_numpy(new_faces).unsqueeze(0).to(device)
        new_bary_coord = torch.from_numpy(np.repeat([[0., 0., 1.]], new_faces.shape[0], axis=0)).unsqueeze(0).to(device)

        ldk_faces_idx = torch.cat((self.full_lmk_faces_idx, new_faces_idx), dim=1) 
        ldk_bary_coord = torch.cat((self.full_lmk_bary_coords, new_bary_coord), dim=1) 

        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         ldk_faces_idx.repeat(bz, 1),
                                         ldk_bary_coord.repeat(bz, 1, 1))

        return landmarks3d.cuda()
    
    def get_3dlandmarks_forehead(self, shape_params, expression_params, euler_angle, trans, arg_focal, cxy):

        vertices = self.forward_geo(shape_params, expression_params, euler_angle, trans, rot=True)

        # New vertices
        # new_faces = np.array([2883, 2875, 2878, 7926, 2865, 7970, 2909, 2912, 2933, 7473, 2411, 161, 3788, 1407, 1459, 2540, 1915, 1328, 1355, 1358, 1361])
        # new_faces = np.array([2875, 7926, 7748, 1293, 1344, 3836, 1431, 1328, 1358]) # manually found in mesh
        new_faces = np.array([1358, 1328, 1431, 3836, 1344, 1293, 7748, 7926, 2875]) # manually found in mesh

        device = shape_params.get_device()
        new_faces_idx = torch.from_numpy(new_faces).unsqueeze(0).to(device)
        new_bary_coord = torch.from_numpy(np.repeat([[0., 0., 1.]], new_faces.shape[0], axis=0)).unsqueeze(0).to(device)
        
        ldk_faces_idx = torch.cat((self.full_lmk_faces_idx, new_faces_idx), dim=1) 
        ldk_bary_coord = torch.cat((self.full_lmk_bary_coords, new_bary_coord), dim=1) 

        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         ldk_faces_idx.repeat(bz, 1),
                                         ldk_bary_coord.repeat(bz, 1, 1))

        return landmarks3d.cuda()
    
    def get_3dlandmarks_all(self, shape_params, expression_params, euler_angle, trans):

        vertices = self.forward_geo(shape_params, expression_params, euler_angle, trans, rot=True)

        # New vertices
        device = shape_params.get_device()
        new_faces_idx = torch.from_numpy(np.arange(0, self.faces_tensor.shape[0], step=1)).unsqueeze(0).to(device)
        new_bary_coord = torch.from_numpy(np.repeat([[0., 0., 1.]], self.faces_tensor.shape[0], axis=0)).unsqueeze(0).to(device)

        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                         new_faces_idx.repeat(bz, 1),
                                         new_bary_coord.repeat(bz, 1, 1))

        return landmarks3d.cuda()

    def forward_geo(self, shape_params, expression_params, euler_angle, trans, rot=True):
        batch_size = shape_params.shape[0]

        eye_pose_params = self.eye_pose.expand(batch_size, -1).cuda()
        betas = torch.cat([shape_params, expression_params[:, :50]], dim=1)

        # TODO: fix FLAME jaw pose
        full_pose = torch.cat(
            [torch.zeros_like(euler_angle),
             self.neck_pose.expand(batch_size, -1),
             expression_params[:, 50:].expand(batch_size, -1),
             eye_pose_params], dim=1)
        
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1).cuda()

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        # Scale to Basel scale
        scale = self.scale_factor.expand(1, 3)
        vertices = torch.mul(vertices, scale).double()

        # Rotate and Translate mesh
        if rot:
            vertices = forward_rott(vertices, euler_angle, trans)

        return vertices.cuda()

    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1).cuda()
        betas = torch.cat([shape_params, expression_params[:, :50]], dim=1)
        full_pose = torch.cat([pose_params[:, :3], self.neck_pose.expand(batch_size, -1), pose_params[:, 3:], eye_pose_params], dim=1)
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1).cuda()

        vertices, _ = lbs(betas, full_pose, template_vertices,
                          self.shapedirs, self.posedirs,
                          self.J_regressor, self.parents,
                          self.lbs_weights, dtype=self.dtype)

        scale = self.scale_factor.expand(1, 3)
        vertices = torch.mul(vertices, scale).double()

        lmk_faces_idx = self.lmk_faces_idx.unsqueeze(dim=0).expand(batch_size, -1)
        lmk_bary_coords = self.lmk_bary_coords.unsqueeze(dim=0).expand(batch_size, -1, -1)
        
        dyn_lmk_faces_idx, dyn_lmk_bary_coords = self._find_dynamic_lmk_idx_and_bcoords(
            full_pose, self.dynamic_lmk_faces_idx.cuda(),
            self.dynamic_lmk_bary_coords.cuda(),
            self.neck_kin_chain.cuda(), dtype=self.dtype)
        lmk_faces_idx = torch.cat([dyn_lmk_faces_idx, lmk_faces_idx], 1)
        lmk_bary_coords = torch.cat([dyn_lmk_bary_coords, lmk_bary_coords], 1)

        landmarks2d = vertices2landmarks(vertices, self.faces_tensor,
                                       lmk_faces_idx,
                                       lmk_bary_coords)
        bz = vertices.shape[0]
        landmarks3d = vertices2landmarks(vertices, self.faces_tensor,
                                       self.full_lmk_faces_idx.repeat(bz, 1),
                                       self.full_lmk_bary_coords.repeat(bz, 1, 1))
        return vertices, landmarks2d, landmarks3d