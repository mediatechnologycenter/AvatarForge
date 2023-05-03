import numpy as np
import open3d as o3d

import argparse
import os
import numpy as np
import torch
from importlib import import_module
from plyfile import PlyData, PlyElement
from skimage import measure
from skimage.draw import ellipsoid


def export_obj(vertices, triangles, diffuse, normals, filename):
    """
    Exports a mesh in the (.obj) format.
    """
    print('Writing to obj...')

    with open(filename, "w") as fh:

        for index, v in enumerate(vertices):
            fh.write("v {} {} {}".format(*v))
            if len(diffuse) > index:
                fh.write(" {} {} {}".format(*diffuse[index]))

            fh.write("\n")

        for n in normals:
            fh.write("vn {} {} {}\n".format(*n))

        for f in triangles:
            fh.write("f")
            for index in f:
                fh.write(" {}//{}".format(index + 1, index + 1))

            fh.write("\n")

    print(f"Finished writing to {filename} with {len(vertices)} vertices")


def export_ply(vertices, diffuse, normals, filename):
    names = 'x, y, z, nx, ny, nz, red, green, blue'
    formats = 'f4, f4, f4, f4, f4, f4, u1, u1, u1'
    arr = np.concatenate((vertices, normals, diffuse * 255), axis = -1)
    vertices_s = np.core.records.fromarrays(arr.transpose(), names=names, formats = formats)

    # Recreate the PlyElement instance
    v = PlyElement.describe(vertices_s, 'vertex')

    # Create the PlyData instance
    p = PlyData([ v ], text = True)

    p.write(filename)


def export_point_cloud(it, ray_origins, ray_directions, depth_fine, dep_target):
    vertices_output = (ray_origins + ray_directions * depth_fine[..., None]).view(-1, 3)
    vertices_target = (ray_origins + ray_directions * dep_target[..., None]).view(-1, 3)
    vertices = torch.cat((vertices_output, vertices_target), dim = 0)
    diffuse_output = torch.zeros_like(vertices_output)
    diffuse_output[:, 0] = 1.0
    diffuse_target = torch.zeros_like(vertices_target)
    diffuse_target[:, 2] = 1.0
    diffuse = torch.cat((diffuse_output, diffuse_target), dim = 0)
    normals = torch.cat((-ray_directions.view(-1, 3), -ray_directions.view(-1, 3)), dim = 0)
    export_obj(vertices, [], diffuse, normals, f"{it:04d}.obj")


def create_point_cloud(ray_origins, ray_directions, depth, color, mask = None):

    if mask is not None:
        ray_directions, depth = ray_directions[mask], depth[mask]

    vertices = (ray_origins + ray_directions * depth[..., None]).view(-1, 3)
    diffuse = color.expand(vertices.shape)
    normals = -ray_directions.view(-1, 3)

    return vertices, diffuse, normals


def extract_iso_level(density, iso_level):
    # Density boundaries
    min_a, max_a, std_a = density.min(), density.max(), density.std()

    # Adaptive iso level
    iso_value = min(max(iso_level, min_a + std_a), max_a - std_a)
    print(f"Min density {min_a}, Max density: {max_a}, Mean density {density.mean()}")
    print(f"Querying based on iso level: {iso_value}")

    return iso_value


def extract_geometry(radiance, pts, iso_level, limit, res):

    # nums = (res,) * 3
    # radiance = radiance.view(*nums, 4).contiguous().numpy()

    radiance = torch.flatten(radiance, end_dim=1)
    pts = torch.flatten(pts, end_dim=1)
    print(radiance.size())
    print(pts.size())

    # Density grid
    density = radiance[..., 3]

    # Adaptive iso level
    iso_value = extract_iso_level(density, iso_level)

    # Extracting iso-surface triangulated
    # Generate a level set about zero of two identical ellipsoids in 3D
    ellip_base = ellipsoid(6, 10, 16, levelset=True)
    ellip_double = np.concatenate((ellip_base[:-1, ...],
                                   ellip_base[2:, ...]), axis=0)

    print(ellip_double.shape)
    results = measure.marching_cubes(density, iso_value)

    # Use contiguous tensors
    vertices, triangles, normals, _ = [torch.from_numpy(np.ascontiguousarray(result)) for result in results]

    # Use contiguous tensors
    normals = torch.from_numpy(np.ascontiguousarray(normals))
    vertices = torch.from_numpy(np.ascontiguousarray(vertices))
    triangles = torch.from_numpy(np.ascontiguousarray(triangles))

    # Normalize vertices, to the (-limit, limit)
    vertices = limit * (vertices / (res / 2.) - 1.)

    return vertices, triangles, normals, density


def export_marching_cubes(radiance, pts, iso_level, limit, res, path):

    mesh_path = os.path.join(path + 'mesh.obj')
    point_cloud_path = os.path.join(path + 'pointcloud.ply')

    vertices, triangles, normals, density = extract_geometry(radiance, pts, iso_level, limit, res)
    diffuse = radiance[..., :3]

    # Export obj
    export_obj(vertices, triangles, diffuse, normals, mesh_path)

    # Export ply
    export_ply(vertices, diffuse, normals, point_cloud_path)


