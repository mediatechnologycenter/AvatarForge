import torch
import os
from model import MattingNetwork
from inference import convert_video

input_video_path = '/home/alberto/data/nerf-videosynth/MTC_data/Clara/Clara.mp4'
checkpoint_path = 'checkpoints/'
results_path = '/home/alberto/data/nerf-videosynth/MTC_data/Clara/matting/'
os.makedirs(results_path, exist_ok=True)

# load model
model = MattingNetwork('mobilenetv3').eval().cuda()  # or "resnet50"
model.load_state_dict(torch.load(checkpoint_path + 'rvm_mobilenetv3.pth'))


convert_video(
    model,                            # The model, can be on any device (cpu or cuda).
    input_source=input_video_path,    # A video file or an image sequence directory.
    output_type='png_sequence',       # Choose "video" or "png_sequence"
    output_composition=results_path,  # File path if video; directory path if png sequence.
    output_video_mbps=4,              # Output video mbps. Not needed for png sequence.
    downsample_ratio=None,            # A hyperparameter to adjust or use None for auto.
    seq_chunk=12,                     # Process n frames at once for better parallelism.
)