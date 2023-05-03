# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
import glob
import os
import subprocess
from typing import Dict

from avatar_backend_api.api_types import InferenceQueueTask
from avatar_backend_api.models.avatar_base_model import AvatarBaseModel
from neural_voice_backend_api.config import NeuralVoiceConfig


class NeuralVoiceModel(AvatarBaseModel):

    def init_model(self):
        for avatar in NeuralVoiceConfig.available_avatars.keys():
            video_input_dir = os.path.join(NeuralVoiceConfig.video_input_dir, avatar)
            features_dir = os.path.join(NeuralVoiceConfig.features_dir, avatar)
            checkpoints_glob = glob.glob(os.path.join(NeuralVoiceConfig.checkpoints_dir, f"*.{avatar}"))
            mappings_glob = glob.glob(os.path.join(NeuralVoiceConfig.mappings_dir, "*"))

            if not os.path.isdir(video_input_dir):
                raise ValueError(f"Could not find directory {video_input_dir}, which is supposed to contain the input video files for avatar {avatar}")

            # Commented for training
            # if not os.path.isdir(features_dir):
            #     raise ValueError(f"Could not find directory {features_dir}, which is supposed to contain the feature files for avatar {avatar}")

            # if not len(checkpoints_glob) >= 1:
            #     raise ValueError(f"Could not find a checkpoints directory for avatar {avatar} under {NeuralVoiceConfig.checkpoints_dir}")

            # if not len(mappings_glob) >= 1:
            #     raise ValueError(f"Could not find a mappings directory for avatar {avatar} under {NeuralVoiceConfig.mappings_dir}")

        # Create input & output directories
        os.makedirs(NeuralVoiceConfig.audio_input_dir, exist_ok=True)
        os.makedirs(NeuralVoiceConfig.features_dir, exist_ok=True)
        os.makedirs(NeuralVoiceConfig.video_input_dir, exist_ok=True)
        os.makedirs(NeuralVoiceConfig.video_output_dir, exist_ok=True)

        print("All files were successfully downloaded, neuralVoice model is ready")

    def inference(self, task: InferenceQueueTask) -> None:
        """ Takes an audio_name, which represents a filename without extension, and an avatar, runs the inference and returns the path to the resulting outpout video file"""

        # Run inference script
        subprocess.run(
            args=["bash", "full_pipeline_nvp.sh", task.request.video_id, task.request.avatar.name, NeuralVoiceConfig.data_base_dir],
            cwd=NeuralVoiceConfig.neural_code_base_dir,  # Working directory from which to run the shell command
            universal_newlines=True,  # Decode output
            check=True,  # Throw exception if return code != 0
        )
        print(f"Inference completed for {task.request.video_id} - {task.request.avatar.value}")

    @staticmethod
    def available_avatars() -> Dict[str, str]:
        with os.scandir(NeuralVoiceConfig.video_input_dir) as iterator:
            return {
                avatar_dir.name: NeuralVoiceConfig.avatar_short_name(avatar_name=avatar_dir.name)
                for avatar_dir in iterator
                if avatar_dir.is_dir()
            }
