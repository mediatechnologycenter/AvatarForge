from numpy.core.numeric import require
from numpy.lib.function_base import quantile
import torch
import numpy as np
from .facemodel import Face_3DMM
from .data_loader import load_dir
from .util import *
from .render_3dmm import Render_3DMM, Render_FLAME
from .FLAME.FLAME import FLAME
from .FLAME.config import cfg
import os
import sys
# import openmesh
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

from pytorch3d.io import IO, save_obj

# def np2mesh(mesh, xnp, path):
#     mesh.points()[:] = xnp
#     openmesh.write_mesh(path, mesh, binary=True)


def track_face_FLAME(dataset_base, h, w, frame_num, out_path, decaexpr_dir, expr_masks_dir):
    '''
    Face tracker using FLAME model.
    Used to have geometry prior for nerf sampling.
    '''

    def set_requires_grad(tensor_list):
        for tensor in tensor_list:
            tensor.requires_grad = True

    start_id = 0
    end_id = frame_num

    id_dir = dataset_base

    debug_ldk_dir = os.path.join(id_dir, 'debug', 'debug_landmarks')
    Path(debug_ldk_dir).mkdir(parents=True, exist_ok=True)
    debug_render_dir = os.path.join(id_dir, 'debug', 'debug_render')
    Path(debug_render_dir).mkdir(parents=True, exist_ok=True)
    debug_meshes_dir = os.path.join(id_dir, 'debug', 'debug_meshes')
    Path(debug_meshes_dir).mkdir(parents=True, exist_ok=True)

    lms, img_paths = load_dir(os.path.join(id_dir, 'landmarks'), os.path.join(id_dir, 'frames'), start_id, end_id)

    num_frames = lms.shape[0]
    cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).cuda()
    # TODO: include jaw pose flame
    id_dim, exp_dim, tex_dim = 100, 53, 50
    model_3dmm = FLAME(cfg.model)

    sel_ids = np.arange(0, num_frames, 40)
    sel_num = sel_ids.shape[0]
    arg_focal = 1600
    arg_landis = 1e5

    for focal in tqdm(range(600, 1500, 100)):
        id_para = lms.new_zeros((1, id_dim), requires_grad=True)
        exp_para = lms.new_zeros((sel_num, exp_dim), requires_grad=True)
        euler_angle = lms.new_zeros((sel_num, 3), requires_grad=True)
        trans = lms.new_zeros((sel_num, 3), requires_grad=True)
        # trans.data[:, 2] -= 1  # DIFFERENT
        trans.data[:, 2] -= 7  # ORIGINAL
        focal_length = lms.new_zeros(1, requires_grad=False)
        focal_length.data += focal
        set_requires_grad([id_para, exp_para, euler_angle, trans])

        optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
        optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=.1)  # Change

        for iter in range(2000):
            id_para_batch = id_para.expand(sel_num, -1)

            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)

            proj_geo = proj_pts(geometry, focal_length, cxy)  # Different: landmarks are already rotated here

            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            loss = loss_lan
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_frame.step()
            if iter % 100 == 0 and False:
                print(focal, 'pose', iter, loss.item())

        for iter in range(2500):
            id_para_batch = id_para.expand(sel_num, -1)

            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)

            proj_geo = proj_pts(geometry, focal_length, cxy)

            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            loss_regid = torch.mean(id_para * id_para)
            loss_regexp = torch.mean(exp_para * exp_para)
            loss = loss_lan + loss_regid * 0.5 + loss_regexp * 0.4
            optimizer_idexp.zero_grad()
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_idexp.step()
            optimizer_frame.step()
            if iter % 100 == 0 and False:
                print(focal, 'poseidexp', iter, loss_lan.item(),
                      loss_regid.item(), loss_regexp.item())
            if iter % 1500 == 0 and iter >= 1500:
                for param_group in optimizer_idexp.param_groups:
                    param_group['lr'] *= 0.2
                for param_group in optimizer_frame.param_groups:
                    param_group['lr'] *= 0.2
        # print(focal, loss_lan.item(), torch.mean(trans[:, 2]).item())

        if loss_lan.item() < arg_landis:
            arg_landis = loss_lan.item()
            arg_focal = focal

    print('find best focal', arg_focal)

    id_para = lms.new_zeros((1, id_dim), requires_grad=True)
    exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
    tex_para = lms.new_zeros((1, tex_dim), requires_grad=True)
    euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
    trans = lms.new_zeros((num_frames, 3), requires_grad=True)
    # trans.data[:, 2] -= 1  # DIFFERENT
    trans.data[:, 2] -= 7  # ORIGINAL

    light_para = lms.new_zeros((num_frames, 27), requires_grad=True)

    focal_length = lms.new_zeros(1, requires_grad=True)
    focal_length.data += arg_focal

    set_requires_grad([id_para, exp_para, tex_para,
                       euler_angle, trans, light_para])

    # DIFFERENT
    # optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.05)
    # optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=.05)

    # ORIGINAL
    optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
    optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=1)

    for iter in tqdm(range(1500)):
        id_para_batch = id_para.expand(num_frames, -1)

        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)

        proj_geo = proj_pts(geometry, focal_length, cxy)  # DIFFERENT

        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms.detach())
        loss = loss_lan
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_frame.step()
        if iter == 1000:
            for param_group in optimizer_frame.param_groups:
                param_group['lr'] = 0.1  # ORIGINAL
                # param_group['lr'] = 0.01  # DIFFERENT
        if iter % 100 == 0 and True:
            print('pose', iter, loss.item())

            # Added save landmarks images for debug
            img = Image.open(img_paths[0])
            colormap_blue = plt.cm.Blues
            colormap_red = plt.cm.Reds

            # plt.imshow(img)
            for num, lin in enumerate(np.linspace(0, 0.9, len(lms[0, :, 0]))):
                plt.scatter(lms[0, num, 0].detach().cpu(),
                            lms[0, num, 1].detach().cpu(),
                            color=colormap_blue(lin),
                            s=10)

                plt.scatter(proj_geo[0, num, 0].detach().cpu(),
                            proj_geo[0, num, 1].detach().cpu(),
                            color=colormap_red(lin),
                            s=10)

            plt.savefig(os.path.join(debug_ldk_dir, f'ldk_1_{iter}.png'))
            plt.close()

    for param_group in optimizer_frame.param_groups:
        param_group['lr'] = 0.1  # ORIGINAL
        # param_group['lr'] = 0.05  # DIFFERENT

    for iter in tqdm(range(2000)):
        id_para_batch = id_para.expand(num_frames, -1)

        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)

        proj_geo = proj_pts(geometry, focal_length, cxy)  # DIFFERENT

        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms.detach())
        loss_regid = torch.mean(id_para * id_para)
        loss_regexp = torch.mean(exp_para * exp_para)
        loss = loss_lan + loss_regid * 0.5 + loss_regexp * 0.4
        optimizer_idexp.zero_grad()
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_idexp.step()
        optimizer_frame.step()
        if iter % 100 == 0 and True:
            print('poseidexp', iter, loss_lan.item(),
                  loss_regid.item(), loss_regexp.item())

            img = Image.open(img_paths[0])
            colormap_blue = plt.cm.Blues
            colormap_red = plt.cm.Reds

            # plt.imshow(img)
            for num, lin in enumerate(np.linspace(0, 0.9, len(lms[0, :, 0]))):
                plt.scatter(lms[0, num, 0].detach().cpu(),
                            lms[0, num, 1].detach().cpu(),
                            color=colormap_blue(lin),
                            s=10)

                plt.scatter(proj_geo[0, num, 0].detach().cpu(),
                            proj_geo[0, num, 1].detach().cpu(),
                            color=colormap_red(lin),
                            s=10)

            plt.savefig(os.path.join(debug_ldk_dir, f'ldk_2_{iter}.png'))
            plt.close()

        if iter % 1000 == 0 and iter >= 1000:
                for param_group in optimizer_idexp.param_groups:
                    param_group['lr'] *= 0.2
                for param_group in optimizer_frame.param_groups:
                    param_group['lr'] *= 0.2

    # THEY DO THIS
    exp_para = exp_para.detach()
    euler_angle = euler_angle.detach()
    trans = trans.detach()
    light_para = light_para.detach()

    batch_size = 10
    device_default = torch.device('cuda:0')
    device_render = torch.device('cuda:0')
    renderer = Render_FLAME(model_3dmm.faces_tensor, arg_focal, h, w, batch_size, device_render)
    
    for i in tqdm(range(int((num_frames-1)/batch_size+1))):
        if (i + 1) * batch_size > num_frames:
            sel_ids = np.arange(num_frames - batch_size, num_frames)
        else:
            sel_ids = np.arange(i * batch_size, i * batch_size + batch_size)
        imgs = []
        for sel_id in sel_ids:
            imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
        imgs = np.stack(imgs)
        sel_imgs = torch.as_tensor(imgs).cuda()

        sel_exp_para = exp_para.new_zeros(
            (batch_size, exp_dim), requires_grad=False)
        sel_exp_para.data = exp_para[sel_ids].clone()
        sel_euler = euler_angle.new_zeros(
            (batch_size, 3), requires_grad=False)
        sel_euler.data = euler_angle[sel_ids].clone()
        sel_trans = trans.new_zeros((batch_size, 3), requires_grad=False)
        sel_trans.data = trans[sel_ids].clone()
        sel_light = light_para.new_zeros(
            (batch_size, 27), requires_grad=False)
        sel_light.data = light_para[sel_ids].clone()

        sel_id_para = id_para.expand(batch_size, -1).detach()

        rott_geo = model_3dmm.forward_geo(sel_id_para, sel_exp_para, sel_euler, sel_trans)

        render_imgs = renderer(rott_geo.to(device_render), model_3dmm.faces_tensor.to(device_render))
        render_imgs = render_imgs.to(device_default)

        mask = (render_imgs[:, :, :, 3]).detach() > 0.0
        mask_img = mask.clone().cpu().numpy() * 255.

        render_proj = sel_imgs.clone()
        render_proj[mask] = render_imgs[mask][..., :3].byte()

        for j in range(sel_ids.shape[0]):
            img_arr = render_proj[j, :, :, :3].byte().detach().cpu().numpy()[
                      :, :, ::-1]
            cv2.imwrite(os.path.join(debug_render_dir, str(sel_ids[j]) + '.jpg'),
                        img_arr)

            cv2.imwrite(os.path.join(expr_masks_dir, str(sel_ids[j]) + '.jpg'),
                        mask_img[j])

            # Save expr
            np.save(os.path.join(decaexpr_dir, f'{sel_ids[j]}.npy'), sel_exp_para[j].cpu().numpy())

            # renderer.get_and_save_mesh(rott_geo, os.path.join(debug_meshes_dir, str(sel_ids[j]) + '.obj'))

    print('about to save params..')

    torch.save({'id': id_para.detach().cpu(), 'exp': exp_para.detach().cpu(),
                'euler': euler_angle.detach().cpu(), 'trans': trans.detach().cpu(),
                'focal': focal_length.detach().cpu(), 'light': light_para.detach().cpu(),
                'text': tex_para.detach().cpu()},
               out_path)

    print('params saved')


def track_face(dataset_base, h, w, frame_num, out_path, expr_masks_dir):
    '''
    Face tracker using partial Basel 2009 model with less vertices.
    '''

    dir_path = os.path.dirname(os.path.realpath(__file__))

    def set_requires_grad(tensor_list):
        for tensor in tensor_list:
            tensor.requires_grad = True

    start_id = 0
    end_id = frame_num

    id_dir = dataset_base
    lms, img_paths = load_dir(os.path.join(id_dir, 'landmarks'), os.path.join(id_dir, 'frames'), start_id, end_id)
    num_frames = lms.shape[0]
    cxy = torch.tensor((w/2.0, h/2.0), dtype=torch.float).cuda()
    id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
    model_3dmm = Face_3DMM(os.path.join(dir_path, '3DMM'),
                           id_dim, exp_dim, tex_dim, point_num)

    sel_ids = np.arange(0, num_frames, 40)
    sel_num = sel_ids.shape[0]
    arg_focal = 1600
    arg_landis = 1e5

    # FINDING BEST FOCAL
    for focal in tqdm(range(600, 1500, 100)):
        id_para = lms.new_zeros((1, id_dim), requires_grad=True)
        exp_para = lms.new_zeros((sel_num, exp_dim), requires_grad=True)
        euler_angle = lms.new_zeros((sel_num, 3), requires_grad=True)
        trans = lms.new_zeros((sel_num, 3), requires_grad=True)
        trans.data[:, 2] -= 7
        focal_length = lms.new_zeros(1, requires_grad=False)
        focal_length.data += focal
        set_requires_grad([id_para, exp_para, euler_angle, trans])

        optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
        optimizer_frame = torch.optim.Adam(
            [euler_angle, trans], lr=.1)

        for iter in range(2000):
            id_para_batch = id_para.expand(sel_num, -1)
            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)
            proj_geo = forward_transform(
                geometry, euler_angle, trans, focal_length, cxy)
            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            loss = loss_lan
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_frame.step()
            if iter % 100 == 0 and False:
                print(focal, 'pose', iter, loss.item())

        for iter in range(2500):
            id_para_batch = id_para.expand(sel_num, -1)
            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)
            proj_geo = forward_transform(
                geometry, euler_angle, trans, focal_length, cxy)
            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            loss_regid = torch.mean(id_para*id_para)
            loss_regexp = torch.mean(exp_para*exp_para)
            loss = loss_lan + loss_regid*0.5 + loss_regexp*0.4
            optimizer_idexp.zero_grad()
            optimizer_frame.zero_grad()
            loss.backward()
            optimizer_idexp.step()
            optimizer_frame.step()
            if iter % 100 == 0 and False:
                print(focal, 'poseidexp', iter, loss_lan.item(),
                      loss_regid.item(), loss_regexp.item())
            if iter % 1500 == 0 and iter >= 1500:
                for param_group in optimizer_idexp.param_groups:
                    param_group['lr'] *= 0.2
                for param_group in optimizer_frame.param_groups:
                    param_group['lr'] *= 0.2
        print(focal, loss_lan.item(), torch.mean(trans[:, 2]).item())

        if loss_lan.item() < arg_landis:
            arg_landis = loss_lan.item()
            arg_focal = focal

    print('find best focal', arg_focal)

    id_para = lms.new_zeros((1, id_dim), requires_grad=True)
    exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=True)
    tex_para = lms.new_zeros((1, tex_dim), requires_grad=True)
    euler_angle = lms.new_zeros((num_frames, 3), requires_grad=True)
    trans = lms.new_zeros((num_frames, 3), requires_grad=True)
    light_para = lms.new_zeros((num_frames, 27), requires_grad=True)
    trans.data[:, 2] -= 7
    focal_length = lms.new_zeros(1, requires_grad=True)
    focal_length.data += arg_focal

    set_requires_grad([id_para, exp_para, tex_para,
                       euler_angle, trans, light_para])

    optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
    optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=1)

    # LANDMARK BASED OPTIMIZATION
    for iter in tqdm(range(1500)):
        id_para_batch = id_para.expand(num_frames, -1)
        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)
        proj_geo = forward_transform(
            geometry, euler_angle, trans, focal_length, cxy)
        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms.detach())
        loss = loss_lan
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_frame.step()
        if iter == 1000:
            for param_group in optimizer_frame.param_groups:
                param_group['lr'] = 0.1
        if iter % 100 == 0 and False:
            print('pose', iter, loss.item())

    for param_group in optimizer_frame.param_groups:
        param_group['lr'] = 0.1

    img = Image.open(img_paths[0])
    # plt.imshow(img)
    plt.scatter(lms[0, :, 0].detach().cpu(), lms[0, :, 1].detach().cpu(), c='r', s=10)
    plt.scatter(proj_geo[0, :, 0].detach().cpu(), proj_geo[0, :, 1].detach().cpu(), c='b', s=10)
    # plt.show()

    for iter in tqdm(range(2000)):
        id_para_batch = id_para.expand(num_frames, -1)
        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)
        proj_geo = forward_transform(
            geometry, euler_angle, trans, focal_length, cxy)
        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms.detach())
        loss_regid = torch.mean(id_para*id_para)
        loss_regexp = torch.mean(exp_para*exp_para)
        loss = loss_lan + loss_regid*0.5 + loss_regexp*0.4
        optimizer_idexp.zero_grad()
        optimizer_frame.zero_grad()
        loss.backward()
        optimizer_idexp.step()
        optimizer_frame.step()
        if iter % 100 == 0 and False:
            print('poseidexp', iter, loss_lan.item(),
                  loss_regid.item(), loss_regexp.item())
        if iter % 1000 == 0 and iter >= 1000:
            for param_group in optimizer_idexp.param_groups:
                param_group['lr'] *= 0.2
            for param_group in optimizer_frame.param_groups:
                param_group['lr'] *= 0.2
    print(loss_lan.item(), torch.mean(trans[:, 2]).item())

    img = Image.open(img_paths[0])
    # plt.imshow(img)
    plt.scatter(lms[0, :, 0].detach().cpu(), lms[0, :, 1].detach().cpu(), c='r', s=10)
    plt.scatter(proj_geo[0, :, 0].detach().cpu(), proj_geo[0, :, 1].detach().cpu(), c='b', s=10)
    # plt.show()

    batch_size = 50

    device_default = torch.device('cuda:0')
    device_render = torch.device('cuda:0')
    renderer = Render_3DMM(arg_focal, h, w, batch_size, device_render)

    sel_ids = np.arange(0, num_frames, int(num_frames/batch_size))[:batch_size]
    imgs = []
    for sel_id in sel_ids:
        imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
    imgs = np.stack(imgs)
    sel_imgs = torch.as_tensor(imgs).cuda()
    sel_lms = lms[sel_ids]
    sel_light = light_para.new_zeros((batch_size, 27), requires_grad=True)
    set_requires_grad([sel_light])
    optimizer_tl = torch.optim.Adam([tex_para, sel_light], lr=.1)
    optimizer_id_frame = torch.optim.Adam(
        [euler_angle, trans, exp_para, id_para], lr=.01)
    
    # RENDERING BASED OPTIMIZATION
    for iter in tqdm(range(71)):
        sel_exp_para, sel_euler, sel_trans = exp_para[sel_ids], euler_angle[sel_ids], trans[sel_ids]
        sel_id_para = id_para.expand(batch_size, -1)
        geometry = model_3dmm.get_3dlandmarks(
            sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy)
        proj_geo = forward_transform(
            geometry, sel_euler, sel_trans, focal_length, cxy)
        loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
        loss_regid = torch.mean(id_para*id_para)
        loss_regexp = torch.mean(sel_exp_para*sel_exp_para)

        sel_tex_para = tex_para.expand(batch_size, -1)
        sel_texture = model_3dmm.forward_tex(sel_tex_para)
        geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
        rott_geo = forward_rott(geometry, sel_euler, sel_trans)
        render_imgs = renderer(rott_geo.to(device_render),
                               sel_texture.to(device_render),
                               sel_light.to(device_render))
        render_imgs = render_imgs.to(device_default)

        mask = (render_imgs[:, :, :, 3]).detach() > 0.0
        render_proj = sel_imgs.clone()
        render_proj[mask] = render_imgs[mask][..., :3].byte()
        loss_col = cal_col_loss(render_imgs[:, :, :, :3], sel_imgs.float(), mask)
        loss = loss_col + loss_lan*3 + loss_regid*2.0 + loss_regexp*1.0
        if iter > 50:
            loss = loss_col + loss_lan*0.05 + loss_regid*1.0 + loss_regexp*0.8
        optimizer_tl.zero_grad()
        optimizer_id_frame.zero_grad()
        loss.backward()
        optimizer_tl.step()
        optimizer_id_frame.step()
        if iter % 50 == 0 and iter >= 5:
            for param_group in optimizer_id_frame.param_groups:
                param_group['lr'] *= 0.2
            for param_group in optimizer_tl.param_groups:
                param_group['lr'] *= 0.2
        #print(iter, loss_col.item(), loss_lan.item(), loss_regid.item(), loss_regexp.item())

    # np2mesh(mesh, geometry[0, ...].detach().cpu().numpy(
    # ), os.path.join(id_dir, 'debug', 'id.ply'))

    light_mean = torch.mean(sel_light, 0).unsqueeze(0).repeat(num_frames, 1)
    light_para.data = light_mean

    exp_para = exp_para.detach()
    euler_angle = euler_angle.detach()
    trans = trans.detach()
    light_para = light_para.detach()

    for i in range(int((num_frames-1)/batch_size+1)):
        if (i+1)*batch_size > num_frames:
            start_n = num_frames-batch_size
            sel_ids = np.arange(num_frames-batch_size, num_frames)
        else:
            start_n = i*batch_size
            sel_ids = np.arange(i*batch_size, i*batch_size+batch_size)
        imgs = []
        for sel_id in sel_ids:
            imgs.append(cv2.imread(img_paths[sel_id])[:, :, ::-1])
        imgs = np.stack(imgs)
        sel_imgs = torch.as_tensor(imgs).cuda()
        sel_lms = lms[sel_ids]

        sel_exp_para = exp_para.new_zeros(
            (batch_size, exp_dim), requires_grad=True)
        sel_exp_para.data = exp_para[sel_ids].clone()
        sel_euler = euler_angle.new_zeros(
            (batch_size, 3), requires_grad=True)
        sel_euler.data = euler_angle[sel_ids].clone()
        sel_trans = trans.new_zeros((batch_size, 3), requires_grad=True)
        sel_trans.data = trans[sel_ids].clone()
        sel_light = light_para.new_zeros(
            (batch_size, 27), requires_grad=True)
        sel_light.data = light_para[sel_ids].clone()

        set_requires_grad([sel_exp_para, sel_euler, sel_trans, sel_light])

        optimizer_cur_batch = torch.optim.Adam(
            [sel_exp_para, sel_euler, sel_trans, sel_light], lr=0.005)

        sel_id_para = id_para.expand(batch_size, -1).detach()
        sel_tex_para = tex_para.expand(batch_size, -1).detach()

        pre_num = 5
        if i > 0:
            pre_ids = np.arange(
                start_n-pre_num, start_n)

        for iter in tqdm(range(50)):
            geometry = model_3dmm.get_3dlandmarks(
                sel_id_para, sel_exp_para, sel_euler, sel_trans, focal_length, cxy)
            proj_geo = forward_transform(
                geometry, sel_euler, sel_trans, focal_length, cxy)
            loss_lan = cal_lan_loss(proj_geo[:, :, :2], sel_lms.detach())
            loss_regexp = torch.mean(sel_exp_para*sel_exp_para)

            sel_geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
            sel_texture = model_3dmm.forward_tex(sel_tex_para)
            geometry = model_3dmm.forward_geo(sel_id_para, sel_exp_para)
            rott_geo = forward_rott(geometry, sel_euler, sel_trans)
            render_imgs = renderer(rott_geo.to(device_render),
                                   sel_texture.to(device_render),
                                   sel_light.to(device_render))
            render_imgs = render_imgs.to(device_default)

            mask = (render_imgs[:, :, :, 3]).detach() > 0.0

            loss_col = cal_col_loss(
                render_imgs[:, :, :, :3], sel_imgs.float(), mask)

            if i > 0:
                geometry_lap = model_3dmm.forward_geo_sub(id_para.expand(
                    batch_size+pre_num, -1).detach(), torch.cat((exp_para[pre_ids].detach(), sel_exp_para)), model_3dmm.rigid_ids)
                rott_geo_lap = forward_rott(geometry_lap,  torch.cat(
                    (euler_angle[pre_ids].detach(), sel_euler)), torch.cat((trans[pre_ids].detach(), sel_trans)))

                loss_lap = cal_lap_loss([rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)],
                                        [1.0])
            else:
                geometry_lap = model_3dmm.forward_geo_sub(
                    id_para.expand(batch_size, -1).detach(), sel_exp_para, model_3dmm.rigid_ids)
                rott_geo_lap = forward_rott(geometry_lap,  sel_euler, sel_trans)
                loss_lap = cal_lap_loss([rott_geo_lap.reshape(rott_geo_lap.shape[0], -1).permute(1, 0)],
                                        [1.0])

            loss = loss_col*0.5 + loss_lan*8 + loss_lap*100000 + loss_regexp*1.0
            if iter > 30:
                loss = loss_col*0.5 + loss_lan*1.5 + loss_lap*100000 + loss_regexp*1.0
            optimizer_cur_batch.zero_grad()
            loss.backward()
            optimizer_cur_batch.step()
            #print(i, iter, loss_col.item(), loss_lan.item(), loss_lap.item(), loss_regexp.item())
        print(str(i) + ' of ' + str(int((num_frames-1)/batch_size+1)) + ' done')
        render_proj = sel_imgs.clone()
        render_proj[mask] = render_imgs[mask][..., :3].byte()
        mask_img = mask.clone().cpu().numpy() * 255.
        debug_render_dir = os.path.join(id_dir, 'debug', 'debug_render')
        debug_meshes_dir = os.path.join(id_dir, 'debug', 'debug_meshes')
        Path(debug_render_dir).mkdir(parents=True, exist_ok=True)
        # Path(debug_meshes_dir).mkdir(parents=True, exist_ok=True)
        for j in range(sel_ids.shape[0]):
            img_arr = render_proj[j, :, :, :3].byte().detach().cpu().numpy()[
                :, :, ::-1]
            cv2.imwrite(os.path.join(debug_render_dir, str(sel_ids[j]) + '.jpg'),
                        img_arr)

            cv2.imwrite(os.path.join(expr_masks_dir, str(sel_ids[j]) + '.jpg'),
                        mask_img[j])

            # renderer.get_and_save_mesh(rott_geo, os.path.join(debug_meshes_dir, str(sel_ids[j]) + '.obj'))

        exp_para[sel_ids] = sel_exp_para.clone()
        euler_angle[sel_ids] = sel_euler.clone()
        trans[sel_ids] = sel_trans.clone()
        light_para[sel_ids] = sel_light.clone()

    torch.save({'id': id_para.detach().cpu(), 'exp': exp_para.detach().cpu(),
                'euler': euler_angle.detach().cpu(), 'trans': trans.detach().cpu(),
                'focal': focal_length.detach().cpu(), 'light': light_para.detach().cpu(),
                'text': tex_para.detach().cpu()}
               , out_path)
    print('params saved')
