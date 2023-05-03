# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os

from face_tracking.data_loader import load_dir
from face_tracking.FLAME.FLAME import FLAME
from face_tracking.FLAME.config import cfg
from face_tracking.render_3dmm import Render_FLAME
from face_tracking.util import *

id_dim, exp_dim, tex_dim = 100, 50, 50


class GeometryGenerator:

    def __init__(self, dataset_base, h, w, frame_num, trackparamspath, mesh_dir):

        self.dataset_base = dataset_base
        self.mesh_dir = mesh_dir

        # Load and unpack tracking params
        params = torch.load(trackparamspath)
        self.id_para = params['id']
        self.exp_para = params['exp']
        self.euler_angle = params['euler']
        self.trans = params['trans']
        self.focal_length = params['focal']
        self.arg_focal = torch.max(self.focal_length).numpy()

        self.light_para = params['light']
        # self.text_para = params['text']

        start_id = 0
        end_id = frame_num
        id_dir = dataset_base
        self.lms, self.img_paths = load_dir(os.path.join(id_dir, 'landmarks'),
                                            os.path.join(id_dir, 'frames'),
                                            start_id, end_id)

        self.num_frames = self.lms.shape[0]
        self.batchsize = 1

        # 3D model and Renderer
        self.model_3dmm = FLAME(cfg.model)

        self.device_render = 'cuda'
        self.renderer = Render_FLAME(self.model_3dmm.faces_tensor,
                                     self.arg_focal,
                                     h, w,
                                     self.batchsize,
                                     self.device_render)

    def generate_mesh(self, index):

        id_para = self.id_para.expand(self.batchsize, -1).cuda()
        exp_para = self.exp_para[index].expand(self.batchsize, -1).cuda()
        euler = self.euler_angle[index].expand(self.batchsize, -1).cuda()
        trans = self.trans[index].expand(self.batchsize, -1).cuda()

        geometry = self.model_3dmm.forward_geo(id_para, exp_para, euler, trans, False)

        save_path = os.path.join(self.mesh_dir, '%5d.obj' % index)

        self.renderer.get_and_save_mesh(geometry, save_path)



