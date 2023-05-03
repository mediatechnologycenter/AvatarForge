import numpy as np
import dlib
import eos
import os
from imageio import imread, imwrite
from skimage.draw import disk
import matplotlib.pyplot as plt
import cv2
import trimesh

PREDICTOR_PATH = 'tracker/eos/shape_predictor_68_face_landmarks.dat'
PATH_TO_EOS = 'tracker/eos/share'

# get viewport matrix
def viewport_matrix(w, h):
    viewport = np.array([0, h, w, -h])

    # scale
    S = np.identity(4, dtype=np.float32)
    S[0][0] *= viewport[2] / 2
    S[1][1] *= viewport[3] / 2
    S[2][2] *= 0.5

    # translate
    T = np.identity(4, dtype=np.float32)
    T[3][0] = viewport[0] + (viewport[2] / 2)
    T[3][1] = viewport[1] + (viewport[3] / 2)
    T[3][2] = 0.5
    return S @ T


def draw_circle(canvas, x, y, r=8, color=(255,255,255)):
    rr,cc = disk((x,y), r, shape=canvas.shape)
    canvas[rr,cc] = color


class EOS_Tracker:
    def __init__(self, path_to_eos, predictor_path):

        # Dlib tracker
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(predictor_path)

        # EOS model
        self.model = eos.morphablemodel.load_model(f"{path_to_eos}/scripts/bfm2017-1_bfm_nomouth.bin")
        self.landmark_mapper = eos.core.LandmarkMapper(f'{path_to_eos}/ibug_to_bfm2017-1_bfm_nomouth.txt')
        self.edge_topology = eos.morphablemodel.load_edge_topology(f'{path_to_eos}/scripts/bfm2017.json')
        self.contour_landmarks = eos.fitting.ContourLandmarks.load(f'{path_to_eos}/ibug_to_bfm2017-1_bfm_nomouth.txt')
        self.model_contour = eos.fitting.ModelContour.load(f'{path_to_eos}/bfm2017-1_bfm_nomouth_model_contours.json')

    def get_facial_landmarks(self, img):
        # get bounding box and facial landmarks
        boxes = self.detector(img)
        lp = []
        for box in boxes:
            shape = self.shape_predictor(img, box)
            index = 1
            landmarks = []
            for part in shape.parts():
                landmarks.append(eos.core.Landmark(str(index), [float(part.y), float(part.x)]))
                index += 1
            lp.append(landmarks)
        return lp

    def fit_shape_and_pose(self, img, landmarks):
        h, w = img.shape[:2]
        mesh, pose, shape_coeffs, blendshape_coeffs = eos.fitting.fit_shape_and_pose(self.model,
                                                                                     landmarks,
                                                                                     self.landmark_mapper,
                                                                                     w,
                                                                                     h,
                                                                                     self.edge_topology,
                                                                                     self.contour_landmarks,
                                                                                     self.model_contour)

        return mesh, pose, shape_coeffs, blendshape_coeffs

    def project_on_img(self, img, mesh, pose):
        canvas = img.copy()
        h, w = img.shape[:2]
        p, mv, vm, fm = self.get_transformation(img, pose)

        # get vertices from mesh
        sampled_verts = mesh.vertices
        for i in sampled_verts:
            tmp = fm @ np.append(i, 1)
            # disregard z and draw 2d pt
            x, y = (int(w / 2 + tmp[0]), int(h / 2 + tmp[1]))
            draw_circle(canvas, x, y)
        return canvas

    def get_transformation(self, img, pose):
        h, w = img.shape[:2]
        vm = viewport_matrix(w, h)

        # get pose and transform to img space
        p = pose.get_projection()  # from world coordinates to view coordinates
        mv = pose.get_modelview()  # From object to world coordinates
        fm = vm @ p @ mv

        return p, mv, vm, fm

    def get_mesh_from_coeffs(self, shape_coeffs, expr_coeffs):
        mesh = self.model.draw_sample(shape_coeffs, expr_coeffs)
        return mesh

    def __call__(self, img, show_on_img=False, show_landmarks=False, save_ply=False):

        # call landmark detector
        landmarks = self.get_facial_landmarks(img)

        if show_landmarks:
            canvas = img.copy()
            for lp in landmarks:
                for point in lp: draw_circle(canvas, point.coordinates[0], point.coordinates[1])
            plt.imshow(canvas)
            plt.show()

        # call eos fitter
        try:
            landmarks = landmarks[0]
            mesh, pose, shape_coeffs, blendshape_coeffs = self.fit_shape_and_pose(img, landmarks)
            # Mesh vertices
            vertices = mesh.vertices
            # print(f'vertices: {len(vertices)}')

            # Triangle vertex indices
            faces = mesh.tvi
            # print(f'faces: {len(faces)}')

            if save_ply:
                mesh_tri = trimesh.Trimesh(vertices=vertices, faces=faces)
                mesh_tri.export('mesh.ply')

            # Get pose
            # rotation = pose.get_rotation()
            # print(f'rotation:\n{rotation}')
            # projection = pose.get_projection()
            # print(f'projection:\n{projection}')

            # print(f'shape_coeffs: {len(shape_coeffs)}')
            # print(f'blendshape_coeffs: {len(blendshape_coeffs)}')

            if show_on_img:
                img_with_mesh = self.project_on_img(img, mesh, pose)
                plt.imshow(img_with_mesh)
                plt.show()

            return mesh, pose, shape_coeffs, blendshape_coeffs

        except IndexError:
            # plt.imshow(img)
            # plt.show()
            imwrite('error.png', img)

            return None, None, np.zeros(100), np.zeros(199)


if __name__ == "__main__":
    img_path = 'IMG_3046.JPG'
    # img = cv2.imread(img_path)
    img = imread(img_path)
    tracker = EOS_Tracker(PATH_TO_EOS, PREDICTOR_PATH)

    mesh, pose, shape_coeffs, blendshape_coeffs = tracker(img)



