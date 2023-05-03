from matplotlib.pyplot import step
import torch
import torch.nn as nn
import numpy as np
import os
from pytorch3d.structures import Meshes
from pytorch3d.renderer import *

import torch.nn.functional as F

from pytorch3d.io import IO, save_obj

from pytorch3d.ops import interpolate_face_attributes

from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)


class SoftSimpleShader(nn.Module):
    """
    Per pixel lighting - the lighting model is applied using the interpolated
    coordinates and normals for each pixel. The blending function returns the
    soft aggregated color using all the faces per pixel.

    To use the default values, simply initialize the shader with the desired
    device e.g.

    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(
            device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)
        return self

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:

        texels = meshes.sample_textures(fragments)
        blend_params = kwargs.get("blend_params", self.blend_params)

        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of SoftPhongShader"
            raise ValueError(msg)
        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 100.0))
        images = softmax_rgb_blend(
            texels, fragments, blend_params, znear=znear, zfar=zfar
        )
        return images


class Render_3DMM(nn.Module):
    def __init__(self, focal=1015, img_h=500, img_w=500, batch_size=1, device=torch.device('cuda:0')):
        super(Render_3DMM, self).__init__()

        self.focal = focal
        self.img_h = img_h
        self.img_w = img_w
        self.device = device
        self.renderer = self.get_render(batch_size)

        dir_path = os.path.dirname(os.path.realpath(__file__))
        topo_info = np.load(os.path.join(
            dir_path, '3DMM', 'topology_info.npy'), allow_pickle=True).item()

        self.tris = torch.as_tensor(topo_info['tris']).to(self.device)
        self.vert_tris = torch.as_tensor(topo_info['vert_tris']).to(self.device)

    def compute_normal(self, geometry):
        vert_1 = torch.index_select(geometry, 1, self.tris[:, 0])
        vert_2 = torch.index_select(geometry, 1, self.tris[:, 1])
        vert_3 = torch.index_select(geometry, 1, self.tris[:, 2])
        nnorm = torch.cross(vert_2-vert_1, vert_3-vert_1, 2)
        tri_normal = nn.functional.normalize(nnorm)
        v_norm = tri_normal[:, self.vert_tris, :].sum(2)
        vert_normal = v_norm / v_norm.norm(dim=2).unsqueeze(2)
        return vert_normal


    def get_render(self, batch_size=1):
        half_s = self.img_w * 0.5
        R, T = look_at_view_transform(10, 0, 0)
        R = R.repeat(batch_size, 1, 1)
        T = torch.zeros((batch_size, 3), dtype=torch.float32).to(self.device)

        cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01, zfar=20,
                                        fov=2*np.arctan(self.img_w//2/self.focal)*180./np.pi)
        lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 1e5]],
            ambient_color=[[1, 1, 1]],
            specular_color=[[0., 0., 0.]],
            diffuse_color=[[0., 0., 0.]]
        )
        sigma = 1e-4
        raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma / 18.0,
            faces_per_pixel=2,
            perspective_correct=False,
        )
        blend_params = blending.BlendParams(background_color=[0, 0, 0])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings,
                cameras=cameras
            ),
            shader=SoftSimpleShader(
                lights=lights,
                blend_params=blend_params,
                cameras=cameras
            ),
        )
        return renderer.to(self.device)

    @staticmethod
    def Illumination_layer(face_texture, norm, gamma):

        n_b, num_vertex, _ = face_texture.size()
        n_v_full = n_b * num_vertex
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8

        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        Y0 = torch.ones(n_v_full).to(gamma.device).float() * a0 * c0
        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(Y0)
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(n_b, num_vertex, 9)
        lighting = Y.bmm(gamma)

        face_color = face_texture * lighting
        return face_color

    def get_mesh(self, rott_geometry):
        mesh = Meshes(rott_geometry, self.tris.float().repeat(
            rott_geometry.shape[0], 1, 1))

        return mesh

    def get_and_save_mesh(self, rott_geometry, save_path):

        verts = rott_geometry.cpu()[0]
        faces = self.tris.float()

        save_obj(save_path, verts=verts, faces=faces)

    def save_mesh(self, mesh, save_path):
        saver = IO()
        saver.save_mesh(mesh, save_path)

    def forward(self, rott_geometry, texture, diffuse_sh):
        face_normal = self.compute_normal(rott_geometry)

        face_color = self.Illumination_layer(texture, face_normal, diffuse_sh)
        face_color = TexturesVertex(face_color)
        mesh = Meshes(rott_geometry, self.tris.float().repeat(
            rott_geometry.shape[0], 1, 1), face_color)

        rendered_img = self.renderer(mesh)
        rendered_img = torch.clamp(rendered_img, 0, 255)

        return rendered_img


class Render_FLAME(nn.Module):
    def __init__(self, faces, focal=1015, img_h=500, img_w=500, batch_size=1, device=torch.device('cuda:0')):
        super(Render_FLAME, self).__init__()

        self.focal = focal
        self.img_h = img_h
        self.img_w = img_w
        self.device = device
        self.renderer = self.get_render(batch_size)

        # dir_path = os.path.dirname(os.path.realpath(__file__))

        self.tris = faces.to(self.device)

    def vertex_normals(self, vertices, faces):
        """
        :param vertices: [batch size, number of vertices, 3]
        :param faces: [batch size, number of faces, 3]
        :return: [batch size, number of vertices, 3]
        """
        assert (vertices.ndimension() == 3)
        assert (faces.ndimension() == 3)
        assert (vertices.shape[0] == faces.shape[0])
        assert (vertices.shape[2] == 3)
        assert (faces.shape[2] == 3)
        bs, nv = vertices.shape[:2]
        bs, nf = faces.shape[:2]
        device = vertices.device
        normals = torch.zeros(bs * nv, 3).to(device)

        faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
        vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

        faces = faces.reshape(-1, 3)
        vertices_faces = vertices_faces.reshape(-1, 3, 3)

        normals.index_add_(0, faces[:, 1].long(),
                           torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1],
                                       vertices_faces[:, 0] - vertices_faces[:, 1]))
        normals.index_add_(0, faces[:, 2].long(),
                           torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2],
                                       vertices_faces[:, 1] - vertices_faces[:, 2]))
        normals.index_add_(0, faces[:, 0].long(),
                           torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0],
                                       vertices_faces[:, 2] - vertices_faces[:, 0]))

        normals = F.normalize(normals, eps=1e-6, dim=1)
        normals = normals.reshape((bs, nv, 3))
        # pytorch only supports long and byte tensors for indexing
        return normals

    def get_render(self, batch_size=1):
        # half_s = self.img_w * 0.5
        R, T = look_at_view_transform(1, 0, 0)
        R = R.repeat(batch_size, 1, 1)
        T = torch.zeros((batch_size, 3), dtype=torch.float32).to(self.device)
        
        cxy = torch.tensor((self.img_w / 2.0, self.img_h / 2.0), dtype=torch.float).unsqueeze(0)
        image_size=torch.tensor((self.img_h, self.img_w)).unsqueeze(0)
        cameras = PerspectiveCameras(device=self.device, R=R, T=T, 
                                     focal_length=self.focal, 
                                     principal_point=cxy,
                                     image_size=image_size, 
                                     in_ndc=False)

        lights = PointLights(
            device=self.device,
            location=[[0.0, 0.0, 0.0]],
        )

        # lights = DirectionalLights(
        #     device=self.device,
        #     direction= [[0.0, 0.0, 1.0]],
        # )

        # lights= AmbientLights(
        #     device=self.device
        # )
        
        raster_settings = RasterizationSettings(
            image_size=(self.img_h, self.img_w),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=raster_settings,
                cameras=cameras
            ),
            shader = SoftPhongShader(
                lights=lights,
                cameras=cameras,
            ),
        )
        return renderer.to(self.device)

    def get_mesh(self, rott_geometry):
        mesh = Meshes(rott_geometry, self.tris.float().repeat(
            rott_geometry.shape[0], 1, 1))

        return mesh

    def get_and_save_mesh(self, rott_geometry, save_path):

        verts = rott_geometry.cpu()[0]
        faces = self.tris.float()

        save_obj(save_path, verts=verts, faces=faces)

    def save_mesh(self, mesh, save_path):
        saver = IO()
        saver.save_mesh(mesh, save_path)

    def forward(self, rott_geometry, faces):

        verts_rgb = (torch.ones_like(rott_geometry)) * 0.5
        
        textures = Textures(verts_rgb=verts_rgb.cuda())
        
        mesh = Meshes(verts=rott_geometry,
                      faces=faces.float().repeat(rott_geometry.shape[0], 1, 1),
                      textures=textures)

        rendered_img = self.renderer(mesh)
        rendered_img = torch.clamp(rendered_img, 0, 255)
        return rendered_img