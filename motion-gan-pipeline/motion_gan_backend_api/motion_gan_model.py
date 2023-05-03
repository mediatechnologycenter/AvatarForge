# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
import os
import subprocess
from enum import Enum
from typing import Dict

from avatar_backend_api.api_types import InferenceQueueTask
from avatar_backend_api.models.avatar_base_model import AvatarBaseModel
from motion_gan_backend_api.config import MotionGanConfig


class Pipeline(Enum):
    value: str
    full_pipeline = "full_pipeline.sh"
    full_pipeline_enhancement = "full_pipeline_enhancement.sh"
    full_pipeline_multiview = "full_pipeline_multiview.sh"


class MotionGANModel(AvatarBaseModel):
    def init_model(self):
        # Checking if either expected files are available
        for avatar in MotionGanConfig.available_avatars.keys():
            checkpoint_path = os.path.join(MotionGanConfig.checkpoints_dir, avatar)
            video_path = os.path.join(MotionGanConfig.video_input_dir, avatar)

            if not os.path.isdir(checkpoint_path):
                if os.path.isdir(video_path):
                    print(f"No checkpoints found for avatar {avatar}. Run inference on it in order to start training it")
                else:
                    raise ValueError(f"Could not find directory {checkpoint_path}, which is supposed to contain the checkpoint files for avatar {avatar}")

        print("All files were successfully downloaded, motionGan model is ready")

    def inference(self, task: InferenceQueueTask, pipeline: Pipeline = Pipeline.full_pipeline) -> None:
        subprocess.run(
            args=["bash", pipeline.value, MotionGanConfig.data_base_dir, task.request.video_id, task.request.avatar.name],
            cwd=MotionGanConfig.motion_gan_base_dir,  # Working directory from which to run the shell command
            universal_newlines=True,  # Decode output
            check=True,  # Throw exception if return code != 0
        )
        print(f"Inference completed for {task.request.video_id} - {task.request.avatar.value}")

    @staticmethod
    def available_avatars() -> Dict[str, str]:
        with os.scandir(MotionGanConfig.checkpoints_dir) as dirs:
            return {
                avatar_dir.name: MotionGanConfig.avatar_short_name(avatar_name=avatar_dir.name)
                for avatar_dir in dirs
                if avatar_dir.is_dir()
            }
