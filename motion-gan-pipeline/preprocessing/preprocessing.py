# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import cv2
from skimage import io
import numpy as np
from tqdm import tqdm
import torch
import json

from autils.options import PreprocessingOptions



class Preprocessor:

    def __init__(self, opt):

        self.opt = opt

        self.step = opt.step

        self.dataroot = opt.dataroot
        self.name = opt.name
        self.target_fps = opt.target_fps
        self.type = opt.preprocessing_type
        self.train_split = opt.train_split
        self.val_split = opt.val_split

        self.dataset_base = os.path.join(self.dataroot, self.name)

        self.mapp = {
            '0': lambda: self.deepspeech_preprocessing(),
            '1': lambda: self.extract_images(),
            '2': lambda: self.landmark_detection(),
            '3': lambda: self.head_pose_estimation(),
            '4': lambda: self.audioexpression_features(),
            '5': lambda: self.face_matting(),
            '6': lambda: self.extract_meshes(),
            '7': lambda: self.save_params(),
            '8': lambda: self.speech_to_text(),
            '9': lambda: self.body_tracking(),
            '10': lambda: self.emotion_detection(),
            '11': lambda: self.edge_detection(),
            '12': lambda: self.audio_noise_reduction(),
            '13': lambda: self.optical_flow(),
        }

    def initialize(self):

        self.audiofeature_dir = os.path.join(self.dataset_base, "audio_feature")

        self.audioexpr_dir = os.path.join(self.dataset_base, 'audio_expr')

        self.decaexpr_dir = os.path.join(self.dataset_base, 'deca_expr')

        self.framesdir = os.path.join(self.dataset_base, 'frames')

        self.landmarkdir = os.path.join(self.dataset_base, 'landmarks')

        self.projectedlankmarkdir = os.path.join(self.dataset_base, 'debug', 'proj_landmarks')

        self.trackparamspath = os.path.join(self.dataset_base, 'track_params.pt')
        self.textpath = os.path.join(self.dataset_base, 'transcript.txt')

        self.mattdir = os.path.join(self.dataset_base, 'matting')

        self.debug_dir = os.path.join(self.dataset_base, 'debug')

        self.mesh_dir = os.path.join(self.dataset_base, 'meshes')

        self.expr_masks_dir = os.path.join(self.dataset_base, 'expr_masks')

        self.body_dir = os.path.join(self.dataset_base, 'body_pose')

        self.emotion_dir = os.path.join(self.dataset_base, 'emotions')

        self.edges_dir = os.path.join(self.dataset_base, 'edges')

        self.cropped_dir = os.path.join(self.dataset_base, 'cropped')

        self.APC_path = os.path.join(self.dataset_base, 'APC_feat.npy')

        self.opticalflow_dir = os.path.join(self.dataset_base, 'opticalflow')
        self.debug_opticalflow_dir = os.path.join(self.dataset_base, 'debug', 'opticalflow')

    def get_valid_frames(self):

        max_frame_num = len(os.listdir(self.framesdir))

        valid_img_ids = []
        for i in range(max_frame_num):
            if os.path.isfile(os.path.join(self.landmarkdir, '%05d.lms' % i)):
                valid_img_ids.append(i)

        valid_img_num = len(valid_img_ids)
        tmp_img = cv2.imread(os.path.join(self.framesdir, '%05d.jpg' % valid_img_ids[0]))
        # tmp_img = cv2.imread(os.path.join(self.framesdir, str(valid_img_ids[0]) + '.png'))

        h, w = tmp_img.shape[0], tmp_img.shape[1]

        self.max_frame_num, self.valid_img_ids, self.valid_img_num, self.h, self.w = \
            max_frame_num, valid_img_ids, valid_img_num, h, w

    def deepspeech_preprocessing(self):
        from deepspeech_features import extract_ds

        print(f'\n\n--- Step 0: Extracting DeepSpeech features ---\n\n')

        os.makedirs(self.audiofeature_dir, exist_ok=True)

        if len(os.listdir(self.audiofeature_dir)) != 0:
            print('Already done.\n\n')

            return

        extract_ds(self.dataset_base, self.name, self.type, self.target_fps)

    def extract_images(self):
        print(f'\n\n--- Step 1: Extracting images from video ---\n\n')

        os.makedirs(self.framesdir, exist_ok=True)

        if len(os.listdir(self.framesdir)) != 0:
            print('Already done.\n\n')

            return
        
        def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
            # initialize the dimensions of the image to be resized and
            # grab the image size
            dim = None
            (h, w) = image.shape[:2]

            # if both the width and height are None, then return the
            # original image
            if width is None and height is None:
                return image

            # check to see if the width is None
            if width is None:
                # calculate the ratio of the height and construct the
                # dimensions
                r = height / float(h)
                dim = (int(w * r), height)

            # otherwise, the height is None
            else:
                # calculate the ratio of the width and construct the
                # dimensions
                r = width / float(w)
                dim = (width, int(h * r))

            # resize the image
            resized = cv2.resize(image, dim, interpolation = inter)

            # return the resized image
            return resized

        vid_file = os.path.join(self.dataset_base, self.name + '.mp4')

        cap = cv2.VideoCapture(vid_file)
        frame_num = 0
        while (True):
            _, frame = cap.read()
            if frame is None:
                break
            # resize frame to be have max dimention = 720
            (h, w) = frame.shape[:2]
            if w > h:
                frame = image_resize(frame, width=720)
            else:
                frame = image_resize(frame, height=720)
                
            cv2.imwrite(os.path.join(self.framesdir, '%05d.jpg' % frame_num), frame)
            frame_num = frame_num + 1
        cap.release()

    def landmark_detection(self):
        import face_alignment

        print('\n\n--- Step 2: Detect Landmarks ---\n\n')

        os.makedirs(self.landmarkdir, exist_ok=True)

        if len(os.listdir(self.landmarkdir)) != 0:
            print('Already done.\n\n')
            return

        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        for image_path in tqdm(os.listdir(self.framesdir)):
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                input = io.imread(os.path.join(self.framesdir, image_path))[:, :, :3]
                preds = fa.get_landmarks(input)
                try:
                    if len(preds) > 0:
                        lands = preds[0].reshape(-1, 2)[:, :2]
                        np.savetxt(os.path.join(self.landmarkdir, image_path[:-3] + 'lms'), lands, '%f')

                except TypeError:
                    pass

    def head_pose_estimation(self):
        from face_tracking.face_tracker import track_face_FLAME, track_face
        from face_tracking.face_tracker_deca import track_face_DECA

        print('\n\n--- Step 3: Estimate Head Pose ---\n\n')
        self.get_valid_frames()

        os.makedirs(self.debug_dir, exist_ok=True)
        os.makedirs(self.expr_masks_dir, exist_ok=True)

        if os.path.isfile(self.trackparamspath):
            print('Already done.\n\n')
            return  

        if self.opt.use_DECA:
            os.makedirs(self.decaexpr_dir, exist_ok=True)

            print('Using DECA tracking..\n')
            track_face_DECA(self.dataset_base, self.h, self.w,
                            self.max_frame_num,
                            self.trackparamspath,
                            self.decaexpr_dir,
                            self.expr_masks_dir)

        elif self.opt.use_FLAME:
            print('Using FLAME tracking..\n')
            track_face_FLAME(self.dataset_base, self.h, self.w,
                             self.max_frame_num,
                             self.trackparamspath,
                             self.decaexpr_dir,
                             self.expr_masks_dir)

        elif self.opt.use_BASEL:
            print('Using BASEL tracking..\n')
            track_face(self.dataset_base,
                       self.h,
                       self.w,
                       self.max_frame_num,
                       self.trackparamspath,
                       self.expr_masks_dir)

    def audioexpression_features(self):
        from third.Audio2ExpressionNet.get_audioexpr import get_audioexpr

        print('\n\n--- Step 4: Extract Audio-Expressions ---\n\n')

        os.makedirs(self.audioexpr_dir, exist_ok=True)

        if len(os.listdir(self.audioexpr_dir)) != 0:
            print('Already done.\n\n')

            return

        get_audioexpr(self.name, self.dataset_base, self.audioexpr_dir)

    def face_matting(self):
        from third.RobustVideoMatting.model import MattingNetwork
        from third.RobustVideoMatting.inference import convert_video

        print('\n\n--- Step 5: Face Matting ---\n\n')

        os.makedirs(self.mattdir, exist_ok=True)

        if len(os.listdir(self.mattdir)) != 0:
            print('Already done.\n\n')
            return

        matting_path = './third/RobustVideoMatting/'

        checkpoint_path = matting_path + 'checkpoints/'
        # vid_file = os.path.join(self.dataset_base, self.name + '.mp4')

        # if not os.path.isfile(vid_file):
        vid_file = self.framesdir

        model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
        model.load_state_dict(torch.load(checkpoint_path + 'rvm_mobilenetv3.pth'))

        convert_video(
            model,  # The model, can be on any device (cpu or cuda).
            input_source=vid_file,  # A video file or an image sequence directory.
            output_type='png_sequence',  # Choose "video" or "png_sequence"
            output_composition=self.mattdir,  # File path if video; directory path if png sequence.
            output_video_mbps=4,  # Output video mbps. Not needed for png sequence.
            downsample_ratio=None,  # A hyperparameter to adjust or use None for auto.
            seq_chunk=12,  # Process n frames at once for better parallelism.
        )

    def extract_meshes(self):
        from facemesh_generator import GeometryGenerator

        print('\n\n--- Step 6: Extract Meshes ---\n\n')
        self.get_valid_frames()

        os.makedirs(self.mesh_dir, exist_ok=True)

        if len(os.listdir(self.mesh_dir)) != 0:
            print('Already done.\n\n')
            return

        generator = GeometryGenerator(self.dataset_base,
                                      self.h, self.w,
                                      self.max_frame_num,
                                      self.trackparamspath,
                                      self.mesh_dir)

        for i in tqdm(range(self.valid_img_num)):
            generator.generate_mesh(i)

    def save_params(self):
        from face_tracking.geo_transform import euler2rot

        print('\n\n--- Step 7: Save Transform Param ---\n\n')
        self.get_valid_frames()

        params_dict = torch.load(self.trackparamspath)
        w, h, valid_img_ids = self.w, self.h, self.valid_img_ids
        focal_len = params_dict['focal']
        euler_angle = params_dict['euler']
        trans = params_dict['trans'] / 10.0  # TODO: why?
        valid_num = euler_angle.shape[0] - 1  # Last element not valid
        print(valid_num, self.valid_img_num)

        train_split = int(self.train_split)
        val_split = int(self.val_split) + train_split

        train_ids = torch.arange(0, train_split)
        val_ids = torch.arange(train_split, val_split)
        test_ids = torch.arange(val_split, valid_num)

        rot = euler2rot(euler_angle)
        rot_inv = rot.permute(0, 2, 1)
        trans_inv = -torch.bmm(rot_inv, trans.unsqueeze(2))

        pose = torch.eye(4, dtype=torch.float32)
        save_ids = ['train', 'val', 'test']
        train_val_ids = [train_ids, val_ids, test_ids]
        mean_z = float(torch.mean(trans_inv[:, 2, 0]).item())
        for i in range(3):
            transform_dict = dict()
            transform_dict['focal_len'] = float(focal_len[0])
            transform_dict['camera_angle_x'] = float(w / 2.0)
            transform_dict['camera_angle_y'] = float(h / 2.0)
            transform_dict['frames'] = []
            ids = train_val_ids[i]
            save_id = save_ids[i]

            for i in tqdm(ids):
                i = i.item()
                frame_dict = dict()

                # image and audio id
                frame_dict['img_id'] = int(valid_img_ids[i])
                frame_dict['aud_id'] = int(valid_img_ids[i])

                # add audio-expression
                try:
                    # add audio_feature
                    frame_dict['audio_feature'] = np.load(os.path.join(self.audiofeature_dir,
                                                                       '%05d.deepspeech.npy' % int(valid_img_ids[i]))
                                                          ).tolist()

                except FileNotFoundError:
                    print('Missing: %05d.deepspeech.npy' % int(valid_img_ids[i]))
                    pass

                # add audio-expression
                try:
                    frame_dict['audio_expr'] = np.load(os.path.join(self.audioexpr_dir,
                                                                    'audio_expression_%05d.npy' % int(valid_img_ids[i]))
                                                       ).tolist()
                except FileNotFoundError:
                    print('Missing: audio_expression_%5d.npy' % int(valid_img_ids[i]))
                    pass

                # add deca-expression
                try:
                    frame_dict['deca_expr'] = np.load(os.path.join(self.decaexpr_dir,
                                                                    '%05d.npy' % i)
                                                       ).tolist()
                except FileNotFoundError:
                    print(f'Missing: %05d.npy' % i)
                    pass

                # add mesh path
                try:
                    frame_dict['mesh_path'] = os.path.join(self.mesh_dir, '%05d.obj' % i)

                except FileNotFoundError:
                    print('Missing: ', os.path.join(self.mesh_dir, '%05d.obj' % i))
                    pass

                # add expression mask
                try:
                    frame_dict['expr_mask'] = os.path.join(self.expr_masks_dir, '%05d.jpg' % i)

                except FileNotFoundError:
                    print('Missing: ', os.path.join(self.expr_masks_dir, '%05d.jpg' % i))
                    pass

                pose[:3, :3] = rot_inv[i]
                pose[:3, 3] = trans_inv[i, :, 0]
                frame_dict['transform_matrix'] = pose.numpy().tolist()

                dir_pose = torch.eye(4, dtype=torch.float32)
                dir_pose[:3, :3] = rot[i]
                dir_pose[:3, 3] = trans[i]
                frame_dict['direct_transform'] = dir_pose.numpy().tolist()

                lms = np.loadtxt(os.path.join(self.landmarkdir, '%05d.lms' % valid_img_ids[i]))
                min_x, max_x = np.min(lms, 0)[0], np.max(lms, 0)[0]

                cx = int((min_x + max_x) / 2.0)
                cy = int(lms[27, 1])

                h_w = int((max_x - cx) * 1.5)
                h_h = int((lms[8, 1] - cy) * 1.15)

                rect_x = cx - h_w
                rect_y = cy - h_h
                if rect_x < 0:
                    rect_x = 0
                if rect_y < 0:
                    rect_y = 0
                rect_w = min(w - 1 - rect_x, 2 * h_w)
                rect_h = min(h - 1 - rect_y, 2 * h_h)

                rect = np.array((rect_x, rect_y, rect_w, rect_h), dtype=np.int32)
                frame_dict['bbox'] = rect.tolist()

                # add dict
                transform_dict['frames'].append(frame_dict)

            with open(os.path.join(self.dataset_base, 'transforms_' + save_id + '.json'), 'w') as fp:
                json.dump(transform_dict, fp, indent=2, separators=(',', ': '))

        near_far = {'near': str(mean_z - 0.2),
                    'far': str(mean_z + 0.4)}

        near_far_json = os.path.join(self.dataset_base, 'near-far.json')
        with open(near_far_json, 'w') as fp:
            json.dump(near_far, fp, indent=2, separators=(',', ': '))

    def speech_to_text(self):
        from preprocessing.speech_to_text import speech_to_text
        print('\n\n--- Step 8: Speech to text ---\n\n')

        if os.path.isfile(self.textpath):
            print('Already done.\n\n')
            return

        speech_to_text(self.name, self.dataset_base, self.textpath)

    def body_tracking(self):
        # from get_densepose import extract_denseposes
        from get_mediapipe import extract_mediapipe

        print('\n\n--- Step 9: Body tracking ---\n\n')

        os.makedirs(self.body_dir, exist_ok=True)

        if len(os.listdir(self.body_dir)) != 0:
            print('Already done.\n\n')
            return

        extract_mediapipe(self.dataset_base, self.body_dir)

    def emotion_detection(self):
        from emoca_tracker import emotion_detection

        print('\n\n--- Step 10: Emotion Detection ---\n\n')
        self.get_valid_frames()

        os.makedirs(self.emotion_dir, exist_ok=True)

        if len(os.listdir(self.emotion_dir)) != 0:
            print('Already done.\n\n')
            return

        print('Using EMOCA emotion tracking..\n')
        emotion_detection(self.dataset_base, 
                          self.emotion_dir)

        return

    def edge_detection(self):
        from edge_creation.utils import make_edges, make_edges_body

        print('\n\n--- Step 11: Edge Detection ---\n\n')
        self.get_valid_frames()

        os.makedirs(self.edges_dir, exist_ok=True)
        os.makedirs(self.cropped_dir, exist_ok=True)

        if len(os.listdir(self.edges_dir)) != 0:
            print('Already done.\n\n')
            return

        ## Switch here for body edges or tracked body pose edges 
        tracked = True

        if not tracked:
            print('Making body edges using extracted edges..')
            make_edges(self.mattdir,
                    self.projectedlankmarkdir,
                    self.edges_dir,
                    self.cropped_dir)

        else:
            # TODO: REMOVE split_val 
            print('Making body edges using tracked body poses..')
            make_edges_body(self.mattdir,
                            self.projectedlankmarkdir,
                            self.edges_dir,
                            self.cropped_dir,
                            self.body_dir,
                            split_val=0.1)

        return
    
    def audio_noise_reduction(self):
        from scipy.io import wavfile
        import noisereduce as nr
        print('\n\n--- Step 12: Noise Reduction ---\n\n')
        print('Skipping for now')
        return
        audio_path = os.path.join(self.dataset_base, self.name + '.wav')
        # load data
        rate, audio = wavfile.read(audio_path)
        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=audio, sr=rate)
        wavfile.write(audio_path, rate, reduced_noise)
        return self
    
    def optical_flow(self):
        from get_optical_flow import compute_optical_flow

        print('\n\n--- Step 13: Optical Flow ---\n\n')
        self.get_valid_frames()

        os.makedirs(self.opticalflow_dir, exist_ok=True)
        os.makedirs(self.debug_opticalflow_dir, exist_ok=True)

        if len(os.listdir(self.opticalflow_dir)) != 0:
            print('Already done.\n\n')
            return

        frames = [os.path.join(self.mattdir, f) for f in sorted(os.listdir(self.mattdir))]

        compute_optical_flow(frames, self.opticalflow_dir, self.debug_opticalflow_dir)
        
        return

    def preprocess_video(self):

        print('Preprocessing video: ', self.name)

        self.mapp[self.step]()

    def preprocess_audio(self):

        print('Preprocessing audio: ', self.name)

        self.mapp[self.step]()

    def __call__(self):
        if self.type == 'audio':
            self.preprocess_audio()

        else:
            self.preprocess_video()


if __name__ == '__main__':

    opt = PreprocessingOptions().parse()
    from pathlib import Path
    opt.name = Path(opt.name).stem
    preprocessor = Preprocessor(opt)
    preprocessor.initialize()
    preprocessor()








