# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import torch

from autils.eos_tracker import *
from third._3DDFA_V2.DDVa_tracker import *
from autils.renderer import Renderer
from pytorch3d.renderer import look_at_view_transform


def call_renderer(mesh, R, T, K=None):
    new_bfm_vertices = torch.FloatTensor(mesh.vertices).unsqueeze(0)
    try:
        bfm_faces_tensor = torch.FloatTensor(mesh.tvi).unsqueeze(0)

    except AttributeError:
        bfm_faces_tensor = torch.FloatTensor(mesh.faces).unsqueeze(0)

    R = torch.FloatTensor(R).unsqueeze(0)
    T = torch.FloatTensor(T).unsqueeze(0)
    if K is not None:
        K = torch.FloatTensor(K).unsqueeze(0)

    print(f'Vertices shape: {new_bfm_vertices.shape}')
    print(f'Faces shape: {bfm_faces_tensor.shape}')
    print(f'R shape: {R.shape}')
    print(f'T shape: {T.shape}')

    # Render with white texture to check if R and T are correct
    mesh_renderer = Renderer(device="cuda" if torch.cuda.is_available() else "cpu", fov=60, aspect_ratio=h/w, image_size=(h, w), R=R, T=T, K=K)
    rendered_np = mesh_renderer.render_mesh(new_bfm_vertices.cuda(), bfm_faces_tensor.cuda())

    return rendered_np


def combine(img, face_render, alpha=0.6):
    final_img = np.array(img)
    face_render = np.array(face_render)
    mask = np.where(np.all(face_render != (255,255,255), axis=-1))
    final_img[mask] = (1-alpha) * final_img[mask] + (alpha) * face_render[mask]
    return final_img

os.environ['PYOPENGL_PLATFORM'] = 'egl'

img_path = 'obama_test.jpeg'
img = imread(img_path)
h, w = img.shape[:2]

print(f'Height: {h}, Width: {w}')
#
# 3DDVA tracker
_3ddva_tracker = DDVA_Tracker()
mesh_tri, R, T, alpha_shp, alpha_exp = _3ddva_tracker(img)
mesh_tri.show()
print(f'\nRotation:\n{R}')
print(f'\nTranslation matrix:\n{T}')


# rendered_np = call_renderer(mesh_tri, R, T)
# plt.imshow(rendered_np)
# plt.savefig('render.png')
# plt.show()
#
# rendered_np = call_renderer(mesh_tri, np.linalg.inv(R), -T)
# plt.imshow(rendered_np)
# plt.savefig('render.png')
# plt.show()

camera = trimesh.scene.cameras.Camera(resolution=(h, w), fov=(63, 63), z_near=0.01, z_far=1000.0)
transform = camera.look_at(mesh_tri.vertices)

R = np.identity(3)
print(f'\nR:\n{R}')

T = np.zeros(3)
print(f'\nT:\n{T}')

rendered_np = call_renderer(mesh_tri, R, T)
plt.imshow(rendered_np)
plt.savefig('render.png')
plt.show()
#
# for i in range(3):
#     R = - R
#     R[i,i] = - R[i,i]
#     rendered_np = call_renderer(mesh_tri, R, T)
#     plt.imshow(rendered_np)
#     plt.savefig('render.png')
#     plt.show()

##########################




