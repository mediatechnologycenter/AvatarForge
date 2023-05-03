import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third/DECA/')))

from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg

from skimage.io import imread, imsave

import numpy as np
from .data_loader import load_dir
from .util import *
from .render_3dmm import Render_FLAME
from .FLAME.FLAME import FLAME
from .FLAME.config import cfg
from .FLAME.lbs import vertices2landmarks
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class DECA_tracker:
    def __init__(self):

        # run DECA
        deca_cfg.model.use_tex = False
        self.deca = DECA(config=deca_cfg, device='cuda')

    def __call__(self, images, tform=None):

        codedict = self.deca.encode(images)

        return codedict


def track_face_DECA(dataset_base, h, w, frame_num, out_path, decaexpr_dir, expr_masks_dir):
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
    debug_mix_dir = os.path.join(id_dir, 'debug', 'debug_mixed')
    Path(debug_mix_dir).mkdir(parents=True, exist_ok=True)
    debug_land_dir = os.path.join(id_dir, 'debug', 'proj_landmarks')
    Path(debug_land_dir).mkdir(parents=True, exist_ok=True)
    debug_land_img_dir = os.path.join(id_dir, 'debug', 'proj_landmarks_img')
    Path(debug_land_img_dir).mkdir(parents=True, exist_ok=True)
    debug_meshes_dir = os.path.join(id_dir, 'debug', 'debug_meshes')
    # Path(debug_meshes_dir).mkdir(parents=True, exist_ok=True)

    lms, img_paths = load_dir(os.path.join(id_dir, 'landmarks'), os.path.join(id_dir, 'frames'), start_id, end_id)
    num_frames = lms.shape[0]
    cxy = torch.tensor((w / 2.0, h / 2.0), dtype=torch.float).cuda()
    id_dim, exp_dim, tex_dim = 100, 53, 50
    model_3dmm = FLAME(cfg.model)

    device_default = torch.device('cuda:0')
    device_render = torch.device('cuda:0')

    deca_tracker = DECA_tracker()
    arg_focal = 600
    # renderer = Render_FLAME(model_3dmm.faces_tensor, arg_focal, h, w, 1, torch.device('cuda:0'))

    # id_para = lms.new_zeros((1, id_dim), requires_grad=True)
    id_para = lms.new_zeros((num_frames, id_dim), requires_grad=True)
    exp_para = lms.new_zeros((num_frames, exp_dim), requires_grad=False)

    # Run deca on all frames
    testdata = datasets.TestData(img_paths, iscrop=True, face_detector='fan')
    for i, data in enumerate(tqdm(testdata)):
        images = data['image'].cuda()[None, ...]
        codedict = deca_tracker(images)
        to_show = codedict['images']
        shape_params = codedict['shape']
        expression_params = codedict['exp']
        pose_params = codedict['pose']

        id_para.data[i] = shape_params
        exp_para.data[i] = torch.cat((expression_params.flatten(), pose_params[:, 3:].flatten()), dim=0)
        np.save(os.path.join(decaexpr_dir, '%05d.npy' % i), exp_para.data[i].unsqueeze(0).cpu().numpy())

        '''# TO TEST DECA
        pose_params = codedict['pose']

        opdict, visdict = deca_tracker.deca.decode(codedict)
        shape_detail_images = visdict['shape_detail_images'][0].permute(1, 2, 0).cpu().numpy()
        shape_detail_images = shape_detail_images * 255.
        shape_detail_images = shape_detail_images.astype(np.uint8)
        img = Image.fromarray(shape_detail_images)

        euler_angle = torch.zeros((1, 3), dtype=torch.double).cuda()
        trans = torch.from_numpy(np.array([[0., 0., -3.]])).double().cuda()
        exp_para_i = torch.cat((expression_params.flatten(), pose_params[:, 3:].flatten()), dim=0).unsqueeze(0)
        
        rott_geo = model_3dmm.forward_geo(shape_params, exp_para_i, euler_angle, trans).float()
        render_imgs = renderer(rott_geo.cuda(), model_3dmm.faces_tensor.cuda())
        render_imgs = render_imgs.cpu().detach().numpy()[0, :, :, :3]
        render_imgs *= 255
        render_imgs = render_imgs.astype(np.uint8)

        # Save landmarks
        im = Image.new('RGB', (w, h), (255, 255, 255))
        geometry = model_3dmm.get_3dlandmarks(
            shape_params, exp_para_i, euler_angle, trans, arg_focal, cxy)

        proj_geo = proj_pts(geometry, arg_focal, cxy)[0]
        for point in proj_geo:
            margin = (max(h, w) // 500) + 1
            ldmks = ([point[0] - margin, point[1] - margin, point[0] + margin, point[1] + margin])
            draw = ImageDraw.Draw(im)
            draw.ellipse(ldmks, fill=(255, 0, 0))

        plt.imshow(im)
        plt.show()

        if i % 500 == 0:
            print(pose_params)
            fig, axs = plt.subplots(1, 3)
            axs[0].imshow(to_show[0].permute(1, 2, 0).cpu().numpy())
            axs[1].imshow(img)
            axs[2].imshow(render_imgs)
            plt.show()
        # '''

    # mean of shape
    id_para = torch.mean(id_para, axis=0).unsqueeze(0)

    # Find best focal
    arg_focal = 600
    arg_landis = 1e5
    sel_ids = np.arange(0, num_frames, 40)
    sel_num = sel_ids.shape[0]
    
    for focal in tqdm(range(700, 1000, 100)):
        
        euler_angle = lms.new_zeros((sel_num, 3), dtype=torch.double, requires_grad=True)
        trans = lms.new_zeros((sel_num, 3), dtype=torch.double, requires_grad=True)
        trans.data[:, 2] -= 1  # DIFFERENT
        # trans.data[:, 2] -= 7  # ORIGINAL
        focal_length = lms.new_zeros(1, requires_grad=False)
        focal_length.data += focal
        set_requires_grad([euler_angle, trans])

        optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=.1)

        for iter in range(2000):
            id_para_batch = id_para.expand(sel_num, -1)
            exp_para_batch = exp_para[sel_ids]
            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para_batch, euler_angle, trans, focal_length, cxy)

            proj_geo = proj_pts(geometry, focal_length, cxy)

            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            loss = loss_lan
            optimizer_frame.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_frame.step()
            if iter % 100 == 0 and False:
                print(focal, 'pose', iter, loss.item())

        for iter in range(2500):
            id_para_batch = id_para.expand(sel_num, -1)
            exp_para_batch = exp_para[sel_ids]

            geometry = model_3dmm.get_3dlandmarks(
                id_para_batch, exp_para_batch, euler_angle, trans, focal_length, cxy)

            proj_geo = proj_pts(geometry, focal_length, cxy)

            loss_lan = cal_lan_loss(
                proj_geo[:, :, :2], lms[sel_ids].detach())
            loss = loss_lan 
            optimizer_frame.zero_grad()
            loss.backward(retain_graph=True)
            optimizer_frame.step()
            if iter % 100 == 0 and False:
                print(focal, 'poseidexp', iter, loss_lan.item(),
                      loss_regid.item(), loss_regexp.item())
            if iter % 1500 == 0 and iter >= 1500:
                for param_group in optimizer_frame.param_groups:
                    param_group['lr'] *= 0.2
        # print(focal, loss_lan.item(), torch.mean(trans[:, 2]).item())

        if loss_lan.item() < arg_landis:
            arg_landis = loss_lan.item()
            arg_focal = focal
    #'''      
    print('find best focal', arg_focal)

    # Free up some memory
    torch.cuda.empty_cache()

    euler_angle = lms.new_zeros((num_frames, 3), dtype=torch.double, requires_grad=True)
    trans = lms.new_zeros((num_frames, 3), dtype=torch.double, requires_grad=True)
    trans.data[:, 2] -= 1  # DIFFERENT
    # trans.data[:, 2] -= 7  # ORIGINAL
    light_para = lms.new_zeros((num_frames, 27), requires_grad=True)

    focal_length = lms.new_zeros(1, requires_grad=False)
    focal_length.data += arg_focal

    set_requires_grad([euler_angle, trans, light_para])

    # ORIGINAL
    # optimizer_idexp = torch.optim.Adam([id_para, exp_para], lr=.1)
    optimizer_frame = torch.optim.Adam([euler_angle, trans], lr=1)

    for iter in tqdm(range(1500)):
        id_para_batch = id_para.expand(num_frames, -1)

        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)

        proj_geo = proj_pts(geometry, focal_length, cxy)

        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms.detach())
        loss = loss_lan
        optimizer_frame.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_frame.step()
        if iter == 1000:
            for param_group in optimizer_frame.param_groups:
                param_group['lr'] = 0.1
        if iter % 250 == 0 and False:
            # print('pose', iter, loss.item())

            img = Image.open(img_paths[0])
            colormap_blue = plt.cm.Blues
            colormap_red = plt.cm.Reds

            plt.imshow(img)
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

    for iter in tqdm(range(2000)):
        id_para_batch = id_para.expand(num_frames, -1)

        geometry = model_3dmm.get_3dlandmarks(
            id_para_batch, exp_para, euler_angle, trans, focal_length, cxy)

        proj_geo = proj_pts(geometry, focal_length, cxy)

        loss_lan = cal_lan_loss(
            proj_geo[:, :, :2], lms.detach())
        loss = loss_lan
        optimizer_frame.zero_grad()
        loss.backward(retain_graph=True)
        optimizer_frame.step()

        if iter % 100 == 0 and False:
            # print('pose', iter, loss_lan.item())

            img = Image.open(img_paths[0])
            colormap_blue = plt.cm.Blues
            colormap_red = plt.cm.Reds

            plt.imshow(img)
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
            for param_group in optimizer_frame.param_groups:
                param_group['lr'] *= 0.2

    # THEY DO THIS
    exp_para = exp_para.detach()
    euler_angle = euler_angle.detach()
    trans = trans.detach()
    light_para = light_para.detach()

    batch_size = 10
    renderer = Render_FLAME(model_3dmm.faces_tensor, arg_focal, h, w, batch_size, device_render)
    pts3D = []

    for i in tqdm(range(int((num_frames - 1) / batch_size + 1))):
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

        rott_geo = model_3dmm.forward_geo(sel_id_para, sel_exp_para, sel_euler, sel_trans).float()
        bz = rott_geo.shape[0]

        sel_pts3D = vertices2landmarks(rott_geo, model_3dmm.faces_tensor,
                                         model_3dmm.full_lmk_faces_idx.repeat(bz, 1),
                                         model_3dmm.full_lmk_bary_coords.repeat(bz, 1, 1).float())
        pts3D.append(sel_pts3D)
        
        ldmks = model_3dmm.get_3dlandmarks_forehead(
                sel_id_para, sel_exp_para, sel_euler, sel_trans, arg_focal, cxy)
        proj_geos = proj_pts(ldmks, focal_length, cxy)
       
        render_imgs = renderer(rott_geo.to(device_render), model_3dmm.faces_tensor.to(device_render))
        render_imgs = render_imgs.to(device_default).detach()
        og_imgs = sel_imgs.clone()

        for j in range(sel_ids.shape[0]):
            # Save render
            img_arr = render_imgs[j, :, :, :3].cpu().numpy()
            img_arr *= 255
            img_arr = img_arr.astype(np.uint8)
            im = Image.fromarray(img_arr)
            im.save(os.path.join(debug_render_dir, '%05d.jpg' % sel_ids[j]))

            # Save mask
            black_pixels_mask = np.all(img_arr != [255, 255, 255], axis=-1)
            mask_img = np.zeros_like(img_arr)
            mask_img[black_pixels_mask] = [255, 255, 255]
        
            im = Image.fromarray(mask_img)
            im.save(os.path.join(expr_masks_dir, '%05d.jpg' % sel_ids[j]))

            # Save mixed
            alpha_blend= 0.1
            og_img = og_imgs[j,:,:,:3].cpu().numpy()
            og_img[black_pixels_mask] = og_img[black_pixels_mask] * alpha_blend + \
                                        img_arr[black_pixels_mask] * (1 - alpha_blend)

            og_img = og_img.astype(np.uint8)
            im = Image.fromarray(og_img)
            im.save(os.path.join(debug_mix_dir, '%05d.jpg' % sel_ids[j]))

            # Save landmarks
            im = Image.new('RGB', (w, h), (255, 255, 255))
            proj_geo = proj_geos[j]
            for point in proj_geo:
                margin = (max(h, w) // 500) + 1
                ldmks = ([point[0] - margin, point[1] - margin, point[0] + margin, point[1] + margin])
                draw = ImageDraw.Draw(im)
                draw.ellipse(ldmks, fill=(255, 0, 0))
                
            im.save(os.path.join(debug_land_img_dir, '%05d.jpg' % sel_ids[j]))
            np.savetxt(os.path.join(debug_land_dir, '%05d.lms' % sel_ids[j]), proj_geo.detach().cpu().numpy())

            
    pts3D = torch.cat(pts3D, dim=0)

    print('about to save params..')

    torch.save({'id': id_para.detach().cpu(), 'exp': exp_para.detach().cpu(),
                'euler': euler_angle.detach().cpu(), 'trans': trans.detach().cpu(),
                'focal': focal_length.detach().cpu(), 'light': light_para.detach().cpu(),
                'pts3D': pts3D.detach().cpu(), 'h': h, 'w': w}, out_path)

    print('params saved')