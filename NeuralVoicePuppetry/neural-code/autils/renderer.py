from typing import Union, List, Tuple

import json
import numpy as np
import torch
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    Textures, TexturesUV
)
from pytorch3d.structures import Meshes


def to_tensor(a, device, dtype=None):
    if isinstance(a, torch.Tensor):
        return a.to(device)
    return torch.from_numpy(np.array(a)).to(device)


class Renderer:
    def __init__(self, device, dist=2, elev=0, azimuth=180, fov=40, image_size=256, aspect_ratio=1, R=None, T=None, K=None, cameras=None):
        # If you provide R and T, you don't need dist, elev, azimuth, fov
        self.device = device

        # Data structures and functions for rendering
        if R is None and T is None:
            R, T = look_at_view_transform(dist, elev, azimuth)
        if cameras is None:
            if K is None:
                cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=1, zfar=10000, aspect_ratio=aspect_ratio,
                                                fov=fov, degrees=True)
            else:
                cameras = FoVOrthographicCameras(device=device, R=R, T=T, K=K)

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            perspective_correct=False
        )

        dist = cameras.T  # Place lights at the same point as the camera
        lights = PointLights(ambient_color=((0.3, 0.3, 0.3),), diffuse_color=((0.7, 0.7, 0.7),), device=device,
                             location=dist)

        # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
        # interpolate the texture uv coordinates for each vertex, sample from a texture image and
        # apply the Phong lighting model
        self.mesh_rasterizer = MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            )
        self._renderer = MeshRenderer(
            rasterizer=self.mesh_rasterizer,
            shader=SoftPhongShader(
                device=device,
                cameras=cameras,
                lights=lights
            )
        )

    def _flatten(self, a):
        return torch.from_numpy(np.array(a)).reshape(-1, 1).to(self.device)

    def render_bfm(self, sp, ep, shape_mu, shape_pcab, expression_pcab, faces, save_to=None):
        sp = self._flatten(sp).double()
        ep = self._flatten(ep).double()
        shape_mu = self._flatten(shape_mu).double()
        shape_pcab = torch.from_numpy(shape_pcab).to(self.device).double()
        expression_pcab = torch.from_numpy(expression_pcab).to(self.device).double()
        faces = torch.from_numpy(np.array(faces)).to(self.device)

        verts_flat = shape_mu + \
                     shape_pcab.mm(sp) + \
                     expression_pcab.mm(ep)
        verts_flat = verts_flat.view(1, -1, 3)
        faces = faces.view(1, -1, 3)
        return self.render_mesh(verts_flat, faces, save_to)

    def render_mesh(self, verts, faces, textures=None, save_to=None, return_format="pil", standardize=False, device="cuda" if torch.cuda.is_available() else "cpu"):
        if textures is not None:
            assert len(textures.shape) == 3, "Please use batches"
            textures = to_tensor(textures, device)
        verts = to_tensor(verts, device, dtype=np.float)
        faces = to_tensor(faces, device, dtype=np.long)

        assert len(verts.shape) == 3, "Please use batches"
        assert len(faces.shape) == 3, "Please use batches"
        bs = verts.shape[0]

        # device
        if verts.device != self.device:
            verts = verts.to(self.device)
        if faces.device != self.device:
            faces = faces.to(self.device)

        # verts = verts / verts.abs().max()
        # standardize
        if standardize:
            verts = verts - verts.mean(1)
            verts = verts / verts.abs().max()

        # Initialize a camera.
        # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
        # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.
        verts = verts.float()
        faces = faces.int()
        verts_rgb = torch.stack([torch.ones_like(verts[i]) for i in range(bs)])  # (1, V, 3)
        if textures is None:
            textures = Textures(verts_rgb=verts_rgb.to(self.device))
        mesh = Meshes(verts=verts, faces=faces, textures=textures)

        images = self._renderer(mesh)
        image = images[0, ..., :3].detach().cpu().numpy()

        image = (image.clip(0, 1) * 255).astype(np.uint8)
        image_pil = Image.fromarray(image)
        if save_to:
            image_pil.save(save_to)
        if return_format.lower() == "pil":
            return image_pil
        elif return_format.lower() == "np":
            return image
        return image


class ImagelessTexturesUV(TexturesUV):
    def __init__(self,
        faces_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        verts_uvs: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor]],
        padding_mode: str = "border",
        align_corners: bool = True,
    ):
        maps = torch.zeros(1, 2,2, 3).to(faces_uvs[0].device)  # This is simply to instantiate a texture, but it is not used.
        super().__init__(maps=maps, faces_uvs=faces_uvs, verts_uvs=verts_uvs, padding_mode=padding_mode, align_corners=align_corners)

    def sample_pixel_uvs(self, fragments, **kwargs) -> torch.Tensor:
        """
        Copied from super().sample_textures and adapted to output pixel_uvs instead of the sampled texture.

        Args:
            fragments:
                The outputs of rasterization. From this we use

                - pix_to_face: LongTensor of shape (N, H, W, K) specifying the indices
                of the faces (in the packed representation) which
                overlap each pixel in the image.
                - barycentric_coords: FloatTensor of shape (N, H, W, K, 3) specifying
                the barycentric coordianates of each pixel
                relative to the faces (in the packed
                representation) which overlap the pixel.

        Returns:
            texels: tensor of shape (N, H, W, K, C) giving the interpolated
            texture for each pixel in the rasterized image.
        """
        if self.isempty():
            faces_verts_uvs = torch.zeros(
                (self._N, 3, 2), dtype=torch.float32, device=self.device
            )
        else:
            packing_list = [
                i[j] for i, j in zip(self.verts_uvs_list(), self.faces_uvs_list())
            ]
            faces_verts_uvs = torch.cat(packing_list)
        texture_maps = self.maps_padded()

        # pixel_uvs: (N, H, W, K, 2)
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )

        N, H_out, W_out, K = fragments.pix_to_face.shape
        N, H_in, W_in, C = texture_maps.shape  # 3 for RGB

        # pixel_uvs: (N, H, W, K, 2) -> (N, K, H, W, 2) -> (NK, H, W, 2)
        pixel_uvs = pixel_uvs.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 2)

        # textures.map:
        #   (N, H, W, C) -> (N, C, H, W) -> (1, N, C, H, W)
        #   -> expand (K, N, C, H, W) -> reshape (N*K, C, H, W)
        texture_maps = (
            texture_maps.permute(0, 3, 1, 2)[None, ...]
            .expand(K, -1, -1, -1, -1)
            .transpose(0, 1)
            .reshape(N * K, C, H_in, W_in)
        )

        # Textures: (N*K, C, H, W), pixel_uvs: (N*K, H, W, 2)
        # Now need to format the pixel uvs and the texture map correctly!
        # From pytorch docs, grid_sample takes `grid` and `input`:
        #   grid specifies the sampling pixel locations normalized by
        #   the input spatial dimensions It should have most
        #   values in the range of [-1, 1]. Values x = -1, y = -1
        #   is the left-top pixel of input, and values x = 1, y = 1 is the
        #   right-bottom pixel of input.

        pixel_uvs = pixel_uvs * 2.0 - 1.0
        return pixel_uvs


class UVRenderer(Renderer):
    def __init__(self, obj_path, device, dist=2, elev=0, azimuth=180, fov=40, image_size=256, R=None, T=None, cameras=None):
        # obj path should be the path to the obj file with the UV parametrization
        super().__init__(device=device, dist=dist, elev=elev, azimuth=azimuth, fov=fov, image_size=image_size, R=R, T=T, cameras=cameras)
        # p = "/media/data/facemodels/basel_facemodel/xucong_bfm/BFM 2017/BFM_face_2017.obj"
        v, f, aux = load_obj(obj_path)
        verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
        faces_uvs = f.textures_idx.to(device)  # (F, 3)
        self.texture = ImagelessTexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs])

    def render(self, meshes):
        # Currently only supports one mesh in meshes
        fragments = self.mesh_rasterizer(meshes)
        rendered_uv = self.texture.sample_pixel_uvs(fragments).detach().cpu().numpy()[0]
        return rendered_uv


class BFMUVRenderer(Renderer):
    def __init__(self, json_path, device, dist=2, elev=0, azimuth=180, fov=40, image_size=256, R=None, T=None, cameras=None):
        # json_path = ".../face12.json"
        super().__init__(device=device, dist=dist, elev=elev, azimuth=azimuth, fov=fov, image_size=image_size, R=R, T=T, cameras=cameras)

        with open(json_path, 'r') as f:
            uv_para = json.load(f)
        verts_uvs = np.array(uv_para['textureMapping']['pointData'])
        faces_uvs = np.array(uv_para['textureMapping']['triangles'])
        verts_uvs = to_tensor(verts_uvs, device).float()
        faces_uvs = to_tensor(faces_uvs, device)
        self.texture = ImagelessTexturesUV(verts_uvs=[verts_uvs], faces_uvs=[faces_uvs])

    def render(self, meshes):
        # Currently only supports one mesh in meshes
        fragments = self.mesh_rasterizer(meshes)
        rendered_uv = self.texture.sample_pixel_uvs(fragments).detach().cpu().numpy()[0]
        return rendered_uv


if __name__ == "__main__":
    height = 256
    path_uv = "face12.json"
    new_bfm_vertices = ...  # vertices of BFM
    bfm_faces_tensor = ...  # triangle faces of BFM

    R, T = look_at_view_transform(eye=((0, 0, 0),), at=((0, 0, -1),), up=((0, 1, 0),))
    uv_renderer = BFMUVRenderer(path_uv, device="cuda" if torch.cuda.is_available() else "cpu", fov=63, image_size=height, R=R, T=T)
    new_bfm_mesh = Meshes(new_bfm_vertices, bfm_faces_tensor)
    rendered_uv_np = uv_renderer.render(new_bfm_mesh)

