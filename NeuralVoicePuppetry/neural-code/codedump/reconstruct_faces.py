# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import tqdm
import torch
import numpy as np
import json
from PIL import Image
from psbody.mesh import Mesh
from autils.voca_helpers import *
from autils.make_video_from_frames import make_video_seq
from third.DECA.decalib.utils.config import cfg
from third.DECA.decalib.utils.renderer import SRenderY
from third.DECA.decalib.models.FLAME import FLAME
from third.DECA.decalib.utils.util import write_obj, batch_orth_proj
from skimage.transform import warp
from skimage.transform import SimilarityTransform
from base_options import FaceReconstructionOptions

def warp_back(image, oldimage, tform):

    alpha = 0.6

    oldimage = oldimage.astype(np.float64) /255.
    new_size = oldimage.shape

    dst_image = warp(image, tform, output_shape=new_size)

    # Mask of non-black pixels.
    mask = np.where(np.all(dst_image <= [0.2, 0.2, 0.2], axis=-1))
    dst_image[mask] = oldimage[mask]

    res = cv2.addWeighted(oldimage, 1 - alpha, dst_image, alpha, 0)
    res = res[:, :, ::-1]

    return res

if __name__ == '__main__':
    opt = FaceReconstructionOptions().parse()

    source_name = opt.source_name
    target_name = opt.target_name
    expression_path = opt.expression_path
    pd_files_path = opt.codedicts_path
    audio_fname = opt.audio_input_dir
    frames_path = opt.frames_path
    tform_path = opt.tform_path

    transfer_name = source_name+'--'target_name

    meshes_out_path = f'results/face_reconstruction/{transfer_name}/meshes'
    os.makedirs(meshes_out_path, exist_ok=True)
    video_out_path = f'results/face_reconstruction/{transfer_name}/video'
    os.makedirs(video_out_path, exist_ok=True)
    images_out_path = f'results/face_reconstruction/{transfer_name}/images'
    os.makedirs(images_out_path, exist_ok=True)
    json_out_path = f'results/face_reconstruction/{transfer_name}/talking_head.json'

    # deca or voca
    mode = 'deca'
    # mode = 'voca'

    # skip saving obj
    skip_obj = True

    # initialize flame
    model_cfg = cfg.model
    flame = FLAME(model_cfg).cuda()
    image_size = cfg.dataset.image_size
    render = SRenderY(image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size).cuda()
    faces = render.faces[0].cpu().numpy()

    # get mean shape
    print('Getting mean shape..')
    pd_files = [f for f in sorted(os.listdir(pd_files_path))
                if os.path.isfile(os.path.join(pd_files_path, f))]

    shapes = torch.tensor([]).cuda()
    codedicts = {}

    for pd_file in tqdm(pd_files):
        codedict = torch.load(os.path.join(pd_files_path, pd_file))
        index = int(pd_file[9:-3])
        codedicts[index] = codedict
        shape = codedict['shape']
        shapes = torch.cat((shapes, shape))

    mean_shape = torch.mean(shapes, dim=0)
    mean_shape = torch.unsqueeze(mean_shape, dim=0)


    # get expressions
    num_expressions = len([name for name in os.listdir(expression_path)])
    sequence_vertices = []

    # make objs
    print('Making meshes..')

    for index in tqdm(range(num_expressions)):

        codedict = codedicts[index]
        expression = np.load(os.path.join(expression_path, f'expr_{index}.npy'))[0]

        pose = codedict['pose'][0, :-3]
        expr = torch.unsqueeze(torch.tensor(expression[:-3], dtype=torch.float32), dim=0).cuda()
        jaw_pose = torch.tensor(expression[-3:], dtype=torch.float32).cuda()
        new_pose = torch.unsqueeze(torch.cat((pose, jaw_pose), dim=0), dim=0)

        # get flame model
        vertices, landmarks2d, landmarks3d = flame(mean_shape, expr, new_pose)

        if not skip_obj:
            filename = os.path.join(meshes_out_path, '%04d.obj'%index)
            write_obj(filename, vertices[0], faces)

        if mode == 'voca':
            # append vertices
            frame = Mesh(filename=filename)
            sequence_vertices.append(frame.v)

        elif mode == 'deca':
            # save images
            trans_verts = batch_orth_proj(vertices, codedict['cam'])
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
            shape_images = render.render_shape(vertices, trans_verts)[0]
            shape_images = shape_images.permute(1,2,0).cpu().numpy()
            shape_images = shape_images * 255.
            shape_images = shape_images.astype(np.uint8)

            filename = os.path.join(images_out_path, '%04d.jpg' % index)

            img = Image.fromarray(shape_images)
            img.save(filename)

    # Render images
    print('Rendering images and making video..')
    if mode == 'voca':
        template = Mesh(sequence_vertices[0], f)
        sequence_vertices = np.stack(sequence_vertices)
        render_sequence_meshes(audio_fname, sequence_vertices, template, images_out_path, video_out_path,
                               uv_template_fname='', texture_img_fname='')

    make_video_seq(audio_fname, images_out_path, video_out_path, 'talking_head.mp4')


    combined_images_out_path = f'results/face_reconstruction/{transfer_name}/combined_images'
    os.makedirs(combined_images_out_path, exist_ok=True)

    tform = np.load(tform_path)

    for image_path, frame in zip(tqdm(sorted(os.listdir(images_out_path))), sorted(os.listdir(frames_path))):

        index = int(image_path[:-4])
        f_tform = SimilarityTransform(matrix=tform[index])

        image = cv2.imread(os.path.join(images_out_path,image_path))
        old_image = cv2.imread(os.path.join(frames_path, frame))

        res = warp_back(image, old_image, f_tform)
        res = res * 255.
        res = res.astype('uint8')
        res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


        file_name = os.path.join(combined_images_out_path, '%04d.jpg' % index)
        cv2.imwrite(file_name, res)

    make_video_seq(audio_fname, combined_images_out_path, video_out_path, 'video_combined.mp4')

