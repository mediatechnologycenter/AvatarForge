# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import torch

from autils.eos_tracker import *
from third._3DDFA_V2.DDVa_tracker import *
from autils.renderer import Renderer
from pytorch3d.renderer import look_at_view_transform
import pytorch3d
from scipy.spatial.transform import Rotation


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
# Launch tracker to get BFM model
eos_tracker = EOS_Tracker(PATH_TO_EOS, PREDICTOR_PATH)
mesh, pose, shape_coeffs, blendshape_coeffs = eos_tracker(img)

# # Store mesh in trimesh
mesh_tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.tvi)

camera = trimesh.scene.cameras.Camera(resolution=(h,w), fov=(63, 63), z_near=0.01, z_far=1000.0)
transform = camera.look_at(mesh_tri.vertices)

# R_trimesh = - transform[0:3, 0:3]
# R_trimesh[1,1] = - R_trimesh[1,1] # flip z to look from other side
# print(f'\nR_trimesh:\n{R_trimesh}')
#
# T_trimesh = transform[0:3, -1]
# print(f'\nT_trimesh:\n{T_trimesh}')

p = pose.get_projection()  # from world coordinates to view coordinates
mv = pose.get_modelview()  # From object to world coordinates
mv_inverse = np.linalg.inv(mv)
rot = pose.get_rotation()
vm = viewport_matrix(w, h)
fm = vm @ p @ mv

v = np.asarray(mesh.vertices)
v = np.hstack([v, np.ones(v.shape[0])])
print(v.shape)
print(v[0])

# # Add rotation
# R_z = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()
# R_y = Rotation.from_euler('xyz', [0, -180, 0], degrees=True).as_matrix()
#
# # R = mv_inverse[0:3, 0:3] @ R_z
# R = mv[0:3, 0:3] @ R_z
# # R = R_pytorch
#
# # T = mv_inverse[0:3, -1] @ R_z
# T = mv[0:3, -1] @ R_z
# # T = T_trimesh

R = np.identity(3)
print(f'\nR:\n{R}')

T = np.zeros(3)
print(f'\nT:\n{T}')

print(f'\nRotation:\n{R}')
print(f'\nTranslation matrix:\n{T}')

rendered_np = call_renderer(mesh, R, T)
plt.imshow(rendered_np)
plt.savefig('render.png')
plt.show()

# v = np.asarray(mesh.vertices)
# R_z = Rotation.from_euler('xyz', [0, 0, 90], degrees=True).as_matrix()
# v2 = v @ mv[0:3,0:3] @ p[0:3,0:3] @ R_z
# plt.figure(figsize=(5,5))
# plt.axis('equal')
# plt.scatter(v2[:,0], v2[:,1], s=0.5)
# plt.savefig('adios.png')
# plt.show()

# ''' try combine function to put mask on img '''
# combined_img = combine(img, rendered_np)
# plt.imshow(combined_img)
# plt.show()

