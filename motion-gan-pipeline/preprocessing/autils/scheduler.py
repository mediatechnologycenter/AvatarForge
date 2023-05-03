import numpy as np
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
import torch
from datetime import datetime
import imageio
from subprocess import call
import trimesh
import open3d as o3d
from .ops3D import export_marching_cubes


class Scheduler:

    def __init__(self, opt):
        self.opt = opt
        self.total_step = 0
        self.val_step = 0
        self.epoch = 0

    def initialize(self, mode='train', num_images=None, num_images_val=None, existing_dir=None):
        self.mode = mode

        if existing_dir is not None:
            self.base_dir = os.path.join(existing_dir, 'test_results')

        else:
            date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
            self.base_dir = os.path.join(self.opt.experiment.outdir, date_time)

        if self.mode == 'train':
            # Setup logging and checkpoints.
            self.num_images = num_images
            self.num_images_val = num_images_val
            self.logdir = os.path.join(self.base_dir, 'logs')
            self.checkpointdir = os.path.join(self.base_dir, 'checkpoints')
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.checkpointdir, exist_ok=True)
            self.writer = SummaryWriter(self.logdir)

            # Write out config parameters.
            with open(os.path.join(self.base_dir, "config.yml"), "w") as f:
                f.write(self.opt.dump())  # cfg, f, default_flow_style=False)

        elif self.mode == 'test':
            # Setup images directory.
            self.savedir = os.path.join(self.base_dir, 'images')
            os.makedirs(self.savedir, exist_ok=True)
            if self.opt.experiment.save_all_images:
                self.disparitydir = os.path.join(self.base_dir, "disparity")
                os.makedirs(self.disparitydir, exist_ok=True)
                self.depthdir = os.path.join(self.base_dir, "depth")
                os.makedirs(self.depthdir, exist_ok=True)
                self.targetdir = os.path.join(self.base_dir, "target")
                os.makedirs(self.targetdir, exist_ok=True)

    def cast_to_image(self, img):
        # Input tensor is (H, W, 3). Convert to (3, H, W).
        img = img.permute(2, 0, 1)

        # Conver to PIL Image and then np.array (output shape: (H, W, 3))
        img = np.array(torchvision.transforms.ToPILImage()(img.detach().cpu()))
        # # Map back to shape (3, H, W), as tensorboard needs channels first.
        # img = np.moveaxis(img, [-1], [0])
        return img

    def cast_one_channel_image(self, tensor):
        img = (tensor - tensor.min()) / (tensor.max() - tensor.min())
        img = img.clamp(0, 1) * 255
        return img.detach().cpu().numpy().astype(np.uint8)

    def write_losses(self, phase, losses, metrics):
        # if self.total_step % self.opt.scheduler.print_every == 0 or \
        #         self.total_step == (self.opt.scheduler.epochs * self.num_images) - 1 or \
        #         phase == 'validation':
        #
        #     if phase == 'validation':
        #         print(f"\n{phase.upper()} ==> Iter: {self.val_step}/{self.num_images_val}" +
        #                        '\n\nLosses - ' + ''.join(f'{i}: {losses[i].item()}, ' for i in losses.keys()))
        #
        #     else:
        #         print(f"\n{phase.upper()} ==> Iter: {self.current_step}/{self.num_images}" +
        #                        '\nLosses - ' + ''.join(f'{i}: {losses[i].item()}, ' for i in losses.keys()))
        #
        #     print('\nMetrics - ' + ''.join(f'{i}: {metrics[i]}, ' for i in metrics.keys()))

        for loss in losses.keys():
            self.writer.add_scalar(f"{phase}/losses/{loss}", losses[loss].item(), self.total_step)

        for metric in metrics.keys():
            self.writer.add_scalar(f"{phase}/metrics/{metric}", metrics[metric], self.total_step)

    def write_images(self, phase, images):
        for nerf in images.keys():
            for image in images[nerf].keys():
                img = images[nerf][image].detach().cpu()
                self.writer.add_image(f'{phase}/{nerf}/{image}', img, self.epoch, dataformats='HWC')

    def save_images(self, rgb, disp, depth, target, image_id):
        savefile = os.path.join(self.savedir, f"{image_id:04d}.png")
        rgb_image = self.cast_to_image(rgb[..., :3])
        imageio.imwrite(savefile, rgb_image)

        if self.opt.experiment.save_all_images:
            # save disp
            savefile = os.path.join(self.disparitydir, f"{image_id:04d}.png")
            disp_image = self.cast_one_channel_image(disp)
            imageio.imwrite(savefile, disp_image)
            # save depth
            savefile = os.path.join(self.depthdir, f"{image_id:04d}.png")
            depth_image = self.cast_one_channel_image(depth)
            imageio.imwrite(savefile, depth_image)
            # save target
            savefile = os.path.join(self.targetdir, f"{image_id:04d}.png")
            target_image = self.cast_to_image(target)
            imageio.imwrite(savefile, target_image)

    def save_video(self, audio_path=None):
        target_fps = 25
        if audio_path is None:
            cmd = f"ffmpeg -framerate {target_fps} -i {self.savedir + '/%04d.png'} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p video.mp4".split()
        else:
            cmd = f"ffmpeg -framerate {target_fps} -i {self.savedir + '/%04d.png'}  -i {audio_path} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p video.mp4".split()

        call(cmd)

    def check_validation(self):
        if self.epoch % self.opt.scheduler.validate_every == 0 or\
           self.epoch == self.opt.scheduler.epochs - 1:
            print("[VAL] =======> Epoch: " + str(self.epoch))
            return True

        else:
            return False

    def check_save(self):
        if self.epoch % self.opt.scheduler.save_every == 0 or \
                self.epoch == self.opt.scheduler.epochs:
            return True

        else:
            return False

    def save_3d_scene(self, vertices, triangles):

        print('Saving mesh ..')
        mesh = trimesh.Trimesh(vertices, triangles)
        trimesh.exchange.export.export_mesh(mesh, os.path.join(self.base_dir, 'mesh.obj'))

    def save_pointcloud(self, vertices):

        print('Saving pointcloud ..')
        mesh = trimesh.Trimesh(vertices)
        trimesh.exchange.export.export_mesh(mesh, os.path.join(self.base_dir, 'pointcloud.ply'))

    def from_pointcloud_to_mesh(self):
        print('Saving mesh from pointcloud ..')

        pcd = o3d.io.read_point_cloud(os.path.join(self.base_dir, 'pointcloud.ply'))
        pcd.estimate_normals()

        # estimate radius for rolling ball
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 4.0 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector([radius, radius * 2]))

        # create the triangular mesh with the vertices and faces from open3d
        tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
                                   vertex_normals=np.asarray(mesh.vertex_normals))

        trimesh.convex.is_convex(tri_mesh)
        trimesh.exchange.export.export_mesh(tri_mesh, os.path.join(self.base_dir, 'mesh.ply'))

