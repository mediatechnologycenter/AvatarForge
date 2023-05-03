# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import numpy as np
import cv2 as cv2
from matplotlib import pyplot as plt
from edge_creation.keypoint2image import interpPoints, drawEdge
import PIL
from PIL import Image, ImageDraw
import torch
from skimage import feature
from logging import raiseExceptions
import torchvision.transforms as transforms
import os
from scipy.ndimage import shift
from tqdm import tqdm
from math import floor


def read_keypoints(A_path, size):        
    # mapping from keypoints to face part 
    part_list = [[list(range(0, 17)) + list(range(68, 83)) + [0]], # face
                    [range(17, 22)],                                  # right eyebrow
                    [range(22, 27)],                                  # left eyebrow
                    [[28, 31], range(31, 36), [35, 28]],              # nose
                    [[36,37,38,39], [39,40,41,36]],                   # right eye
                    [[42,43,44,45], [45,46,47,42]],                   # left eye
                    [range(48, 55), [54,55,56,57,58,59,48]],          # mouth
                    [range(60, 65), [64,65,66,67,60]]                 # tongue
                ]
    label_list = [50, 100, 100, 125, 150, 150, 200, 250] # labeling for different facial parts        
    keypoints = np.loadtxt(A_path)[:,:2]
    # keypoints = np.loadtxt(A_path, delimiter=' ')
    
    # add upper half face by symmetry
    pts = keypoints[:17, :].astype(np.int32)
    baseline_y = (pts[0,1] + pts[-1,1]) / 2
    upper_pts = pts[1:-1,:].copy()
    upper_pts[:,1] = baseline_y + (baseline_y-upper_pts[:,1]) * 2 // 3
    keypoints = np.vstack((keypoints, upper_pts[::-1,:]))  

    # label map for facial part
    w, h = size
    part_labels = np.zeros((h, w), np.uint8)
    for p, edge_list in enumerate(part_list):                
        indices = [item for sublist in edge_list for item in sublist]
        pts = keypoints[indices, :].astype(np.int32)
        cv2.fillPoly(part_labels, pts=[pts], color=label_list[p]) 

    return keypoints, part_list, part_labels

def read_keypoints_forehead(A_path, size):        
    # mapping from keypoints to face part 
    part_list = [[list(range(0, 17)) + list(range(68, 77)) + [0]], # face
                    [range(17, 22)],                                  # right eyebrow
                    [range(22, 27)],                                  # left eyebrow
                    [[28, 31], range(31, 36), [35, 28]],              # nose
                    [[36,37,38,39], [39,40,41,36]],                   # right eye
                    [[42,43,44,45], [45,46,47,42]],                   # left eye
                    [range(48, 55), [54,55,56,57,58,59,48]],          # mouth
                    [range(60, 65), [64,65,66,67,60]]                 # tongue
                ]
    label_list = [50, 100, 100, 125, 150, 150, 200, 250] # labeling for different facial parts        
    keypoints = np.loadtxt(A_path)[:,:2]
    
    # label map for facial part
    w, h = size
    part_labels = np.zeros((h, w), np.uint8)
    for p, edge_list in enumerate(part_list):                
        indices = [item for sublist in edge_list for item in sublist]
        pts = keypoints[indices, :].astype(np.int32)
        cv2.fillPoly(part_labels, pts=[pts], color=label_list[p]) 

    return keypoints, part_list, part_labels

def draw_face_edges(keypoints, part_list, size):
    w, h = size
    edge_len = 3  # interpolate 3 keypoints to form a curve when drawing edges
    # edge map for face region from keypoints
    im_edges = np.zeros((h, w), np.uint8) # edge map for all edges
    dist_tensor = 0
    e = 1                
    for edge_list in part_list:
        for edge in edge_list:
            for i in range(0, max(1, len(edge)-1), edge_len-1): # divide a long edge into multiple small edges when drawing
                sub_edge = edge[i:i+edge_len]
                x = keypoints[sub_edge, 0]
                y = keypoints[sub_edge, 1]
                                
                curve_x, curve_y = interpPoints(x, y) # interp keypoints to get the curve shape                    
                drawEdge(im_edges, curve_x, curve_y)

    return im_edges, dist_tensor

def draw_body_edges(keypoints, size):
    w, h = size
    body_edges = Image.fromarray(np.zeros((h, w), np.uint8)) # edge map for all edges

    # left to right shoulder
    a = keypoints[11][:-2]# left shoulder
    b = keypoints[12][:-2] # right shoulder

    # create line image
    tmp1 = ImageDraw.Draw(body_edges)  
    shape = [a[0], a[1], b[0], b[1]]
    tmp1.line(shape, fill=(255), width = 3)
    
    # mid to nose
    c = keypoints[0][:-2]
    d = (a + b) / 2
    shape = [c[0], c[1], d[0], d[1]]
    tmp1.line(shape, fill=(255), width = 3)

    # left to right hips
    e = keypoints[23][:-2]# left hip
    f = keypoints[24][:-2] # right hip

    # create line image
    shape = [a[0], a[1], e[0], e[1]]
    tmp1.line(shape, fill=(255), width = 3)
    shape = [b[0], b[1], f[0], f[1]]
    tmp1.line(shape, fill=(255), width = 3)

    # Cross for content
    shape = [a[0], a[1], f[0], f[1]]
    tmp1.line(shape, fill=(255), width = 3)
    shape = [b[0], b[1], e[0], e[1]]
    tmp1.line(shape, fill=(255), width = 3)

    return body_edges

def draw_keypoints(img_path, keypoints, color=(255, 0, 0)):
    tmp = Image.open(img_path)
    size = tmp.size

    for point in keypoints:
        margin = (max(size) // 500) + 3
        ldmks = ([point[0] - margin, point[1] - margin, point[0] + margin, point[1] + margin])
        draw = ImageDraw.Draw(tmp)
        draw.ellipse(ldmks, fill=color)
    
    return tmp

def get_crop_coords(keypoints, size):                
    min_y, max_y = keypoints[:,1].min(), keypoints[:,1].max()
    min_x, max_x = keypoints[:,0].min(), keypoints[:,0].max()                
    xc = (min_x + max_x) // 2
    yc = (min_y*3 + max_y) // 4
    h = w = (max_x - min_x) * 2.5        
    xc = min(max(0, xc - w//2) + w, size[0]) - w//2
    yc = min(max(0, yc - h//2) + h, size[1]) - h//2
    min_x, max_x = xc - w//2, xc + w//2
    min_y, max_y = yc - h//2, yc + h//2        
    return [int(min_y), int(max_y), int(min_x), int(max_x)]        

def crop(img, ext_values):
    if isinstance(img, np.ndarray):
        return img[ext_values[0]:ext_values[1], ext_values[2]:ext_values[3]]
    else:
        return img.crop((ext_values[2], ext_values[0], ext_values[3], ext_values[1]))

def get_img_params(size, loadSize=320):
    w, h = size
    new_h, new_w = h, w                

    new_w = loadSize
    new_h = loadSize * h // w

    new_w = int(round(new_w / 4)) * 4
    new_h = int(round(new_h / 4)) * 4         

    return {'new_size': (new_w, new_h)}
  
def get_transform(params, method=Image.BICUBIC, normalize=True, toTensor=True):
    transform_list = []
    ### resize input image
    transform_list.append(transforms.Lambda(lambda img: __scale_image(img, params['new_size'], method)))

    if toTensor:
        transform_list += [transforms.ToTensor()]
    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_image(img, size, method=Image.BICUBIC):
    w, h = size    
    return img.resize((w, h), method)


def convert_to_rgb(image):
        if image.mode == 'RGBA':
            image.load() 
            image_new = Image.new("RGB", image.size, (255, 255, 255))
            image_new.paste(image, mask=image.split()[3])
        elif image.mode == 'RGB':
            image_new = image
        else:
            raiseExceptions('Non-compatible image format!')
        return image_new

def get_edge_predicted(img_num, out_dir, landmark_name, predicted_ldk, im_size):
    # Used at inference time when combining generated landmarks and predicted shoulders

    params = get_img_params(im_size, loadSize=np.max(im_size))      
    transform_scaleA = get_transform(params, method=Image.BILINEAR, normalize=False, toTensor=False)

    # Draw face
    # keypoints, part_list, part_labels = read_keypoints(landmark_name, im_size)
    keypoints, part_list, part_labels = read_keypoints_forehead(landmark_name, im_size)
    im_edges, dist_tensor = draw_face_edges(keypoints, part_list, im_size)

    # body edges
    w, h = im_size
    body_edges = Image.fromarray(np.zeros((h, w), np.uint8)) # edge map for all edges

    # left to right shoulder
    a = predicted_ldk[0] # left shoulder
    b = predicted_ldk[1] # right shoulder

    # create line image
    tmp1 = ImageDraw.Draw(body_edges)  
    shape = [a[0], a[1], b[0], b[1]]
    tmp1.line(shape, fill=(255), width = 3)
    
    # mid to nose
    c = keypoints[33][:2]
    d = (a + b) / 2
    shape = [c[0], c[1], d[0], d[1]]
    tmp1.line(shape, fill=(255), width = 3)

    # left to right hips
    e = predicted_ldk[2]# left hip
    f = predicted_ldk[3] # right hip

    # create line image
    shape = [a[0], a[1], e[0], e[1]]
    tmp1.line(shape, fill=(255), width = 3)
    shape = [b[0], b[1], f[0], f[1]]
    tmp1.line(shape, fill=(255), width = 3)

    # Cross for content
    shape = [a[0], a[1], f[0], f[1]]
    tmp1.line(shape, fill=(255), width = 3)
    shape = [b[0], b[1], e[0], e[1]]
    tmp1.line(shape, fill=(255), width = 3)

    # clean
    body_edges = np.array(body_edges) * (part_labels == 0) # remove edges within face
    im_edges += body_edges

    # Transform and save
    edge_path = os.path.join(out_dir, 'edges')
    os.makedirs(edge_path, exist_ok=True)
    edge_image = transform_scaleA(Image.fromarray(im_edges))
    edge_image.save(os.path.join(edge_path, '%05d.png' % img_num))

def get_edge_image_mixed(out_dir, img_name, landmark_name, tr_landmark_name, crop_coords, img_num):
    # Used at inference time when combining generated landmarks and original frame's edges

    # Flag for cropping
    crop_flag = False

    # Make edges from generated landmarks and closest image
    img = Image.open(img_name)
    img = convert_to_rgb(img)
    img_size = img.size
    points = np.loadtxt(landmark_name)[:,:2]
    tr_points = np.loadtxt(tr_landmark_name, delimiter=' ')
    face_dist_x = np.mean(tr_points[31:36, 0]-points[31:36, 0])
    face_dist_y = np.mean(tr_points[31:36, 1]-points[31:36, 1])

    if crop_flag:
        params = get_img_params(crop(img, crop_coords).size) 

    else:
        params = get_img_params(img.size, loadSize=np.max(img.size))        

    transform_scaleA = get_transform(params, method=Image.BILINEAR, normalize=False, toTensor=False)
    transform_label = get_transform(params, method=Image.NEAREST, normalize=False, toTensor=False)
    transform_scaleB = get_transform(params, normalize=False, toTensor=False)

    # keypoints, part_list, part_labels = read_keypoints(landmark_name, img_size)
    keypoints, part_list, part_labels = read_keypoints_forehead(landmark_name, img_size)

    im_edges, dist_tensor = draw_face_edges(keypoints, part_list, img_size)

    edges = feature.canny(shift(np.array(img.convert('L')), (-face_dist_y, -face_dist_x), mode='nearest'))        
    # edges = feature.canny(np.array(img.convert('L')))        
    edges = edges * (part_labels == 0)  # remove edges within face
    im_edges += (edges * 255).astype(np.uint8)

    if crop_flag:
        edge_image = transform_scaleA(Image.fromarray(crop(im_edges, crop_coords)))
        label_image = transform_label(Image.fromarray(crop(part_labels.astype(np.uint8), crop_coords)))
        img_image = transform_scaleB(crop(img, crop_coords))
    
    else:
        edge_image = transform_scaleA(Image.fromarray(im_edges))
        label_image = transform_label(Image.fromarray(part_labels.astype(np.uint8)))
        img_image = transform_scaleB(img)

    edge_path = os.path.join(out_dir, 'edges')
    os.makedirs(edge_path, exist_ok=True)
    edge_image.save(os.path.join(edge_path, '%05d.png' % img_num))

def get_edge_image(index, img_path, tracked_landmark_path, edges_dir, cropped_dir, im_size, crop_coords):
    # Used in preprocessing when combining tracked landmarks with frame's edges

    # Flag for cropping
    crop_flag = False

    # Load image
    img = Image.open(img_path)
    img = convert_to_rgb(img)
    
    # Make crop parameters
    if crop_flag:
        params = get_img_params(crop(img, crop_coords).size)        
    
    else:
        params = get_img_params(img.size, loadSize=np.max(img.size))      

    transform_scaleA = get_transform(params, method=Image.BILINEAR, normalize=False, toTensor=False)
    transform_scaleB = get_transform(params, normalize=False, toTensor=False)

    # Draw face
    # keypoints, part_list, part_labels = read_keypoints(tracked_landmark_path, im_size)
    keypoints, part_list, part_labels = read_keypoints_forehead(tracked_landmark_path, im_size)
    im_edges, dist_tensor = draw_face_edges(keypoints, part_list, im_size)

    # Other Edges
    edges = feature.canny(np.array(img.convert('L')))        
    edges = edges * (part_labels == 0)  # remove edges within face
    im_edges += (edges * 255).astype(np.uint8)

    # Transform
    if crop_flag:
        edge_image = transform_scaleA(Image.fromarray(crop(im_edges, crop_coords)))
        img_image = transform_scaleB(crop(img, crop_coords))

    
    else:
        edge_image = transform_scaleA(Image.fromarray(im_edges))
        img_image = transform_scaleB(img)

    # Save edges and cropped target for GAN training
    edge_image.save(os.path.join(edges_dir, '%05d.png' % index))
    img_image.save(os.path.join(cropped_dir, '%05d.png' % index))

def get_edge_image_body(index, img_path, tracked_landmark_path, edges_dir, cropped_dir, im_size, crop_coords, body_landmark_path):
    # Used in preprocessing when combining tracked landmarks with tracked body edges

    # Flag for cropping
    crop_flag = False

    # Load image
    img = Image.open(img_path)
    img = convert_to_rgb(img)
    
    # Make crop parameters
    if crop_flag:
        params = get_img_params(crop(img, crop_coords).size)        
    
    else:
        params = get_img_params(img.size, loadSize=np.max(img.size))      

    transform_scaleA = get_transform(params, method=Image.BILINEAR, normalize=False, toTensor=False)
    transform_scaleB = get_transform(params, normalize=False, toTensor=False)

    # Draw face
    # keypoints, part_list, part_labels = read_keypoints(tracked_landmark_path, im_size)
    keypoints, part_list, part_labels = read_keypoints_forehead(tracked_landmark_path, im_size)
    im_edges, dist_tensor = draw_face_edges(keypoints, part_list, im_size)

    # body edges
    body_keypoints = np.load(body_landmark_path)
    body_edges = np.array(draw_body_edges(body_keypoints, im_size))
    body_edges = body_edges * (part_labels == 0) # remove edges within face
    im_edges += body_edges

    # Save landmarks for DEBUGGING
    debug = False
    if debug:
        debug_dir = os.path.join(edges_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, '%05d.png' % index)
        tmp1 = draw_keypoints(img_path, keypoints, color=(255, 0, 0))
        Image.fromarray(np.array(tmp1)).save(debug_path)
        tmp2 = draw_keypoints(debug_path, body_keypoints, color=(0, 0, 255))
        Image.fromarray(np.array(tmp2)).save(debug_path)
    
    # Transform
    if crop_flag:
        edge_image = transform_scaleA(Image.fromarray(crop(im_edges, crop_coords)))
        img_image = transform_scaleB(crop(img, crop_coords))

    
    else:
        edge_image = transform_scaleA(Image.fromarray(im_edges))
        img_image = transform_scaleB(img)

    # Save edges and cropped target for GAN training
    edge_image.save(os.path.join(edges_dir, '%05d.png' % index))
    img_image.save(os.path.join(cropped_dir, '%05d.png' % index))

def make_edges(mattdir, landmarkdir, edges_dir, cropped_dir):
    
    # Compute crop size
    img = Image.open(os.path.join(mattdir, '%05d.png' % 0))
    im_size = img.size

    points = np.mean([np.loadtxt(os.path.join(landmarkdir, ldk))[:,:2] for ldk in os.listdir(landmarkdir)], axis=0)
    crop_coords = get_crop_coords(points, im_size)

    for index in tqdm(range(len(os.listdir(mattdir)))):

        img_path = os.path.join(mattdir, '%05d.png' % index)
        tracked_landmark_path = os.path.join(landmarkdir, '%05d.lms' % index)

        get_edge_image(index, img_path, tracked_landmark_path, edges_dir, cropped_dir, im_size, crop_coords)

def make_edges_body(mattdir, landmarkdir, edges_dir, cropped_dir, body_dir, split_val=0.0):
    # Compute crop size
    img = Image.open(os.path.join(mattdir, '%05d.png' % 0))
    im_size = img.size

    points = np.mean([np.loadtxt(os.path.join(landmarkdir, ldk))[:,:2] for ldk in os.listdir(landmarkdir)], axis=0)
    crop_coords = get_crop_coords(points, im_size)

    num_frames = len(os.listdir(mattdir))
    num_train = floor(num_frames * (1.0 - split_val))

    # train
    for index in tqdm(range(num_train)):

        img_path = os.path.join(mattdir, '%05d.png' % index)
        tracked_landmark_path = os.path.join(landmarkdir, '%05d.lms' % index)
        body_landmark_path = os.path.join(body_dir, '%05d.npy' % index)

        get_edge_image_body(index, img_path, tracked_landmark_path, edges_dir, cropped_dir, im_size, crop_coords, body_landmark_path)
    
    # val
    for index in tqdm(range(num_train, num_frames)):

        img_path = os.path.join(mattdir, '%05d.png' % index)
        tracked_landmark_path = os.path.join(landmarkdir, '%05d.lms' % index)
        body_landmark_path = os.path.join(body_dir, '%05d.npy' % index)

        os.makedirs(edges_dir + '_val', exist_ok=True)
        os.makedirs(cropped_dir + '_val', exist_ok=True)

        get_edge_image_body(index, img_path, tracked_landmark_path, edges_dir + '_val', cropped_dir + '_val', im_size, crop_coords, body_landmark_path)
    
