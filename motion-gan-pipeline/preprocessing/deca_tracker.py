# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third/DECA/')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg


class DECA_tracker:
    def __init__(self, video_path):

        # load test images
        self.testdata = datasets.TestData(video_path, iscrop=True, face_detector='fan')
        self.device = 'cuda'

        # run DECA
        deca_cfg.model.use_tex = False
        self.deca = DECA(config=deca_cfg, device=self.device)

    def get_flame_mesh(self, codedict):

        verts, landmarks2d, landmarks3d = self.deca.flame(shape_params=codedict['shape'],
                                                          expression_params=codedict['exp'],
                                                          pose_params=codedict['pose'])

        trans_verts = util.batch_orth_proj(verts, codedict['cam'])
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        return verts, trans_verts

    def save_obj(self, filename, vertices):
        faces = self.deca.render.faces[0].cpu().numpy()

        util.write_obj(filename, vertices, faces)

    def __call__(self, images, tform=None):

        codedict = self.deca.encode(images)

        return codedict
