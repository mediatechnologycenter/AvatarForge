# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import shutil

from avatar_backend_api.background_tools.inference_queue import InferenceQueue, InferenceQueueTask
from avatar_backend_api.clients.io_client import VIDEO_EXTENSION

from neural_voice_backend_api.config import NeuralVoiceConfig


class NeuralVoiceInferenceQueue(InferenceQueue):

    def post_processing(self, task: InferenceQueueTask) -> None:
        # Delete input audio
        self.io_client.audio.delete_file(video_id=task.request.video_id)

        # Delete features
        features_path = os.path.join(NeuralVoiceConfig.features_dir, task.request.video_id)
        shutil.rmtree(path=features_path, ignore_errors=True)

        # Rename file and return video path
        if not NeuralVoiceConfig.mockBackend:
            self.io_client.video.rename_file(
                filename=f"{task.request.video_id}_to_{task.request.avatar.name}{VIDEO_EXTENSION}",
                new_filename=task.request.video_id + VIDEO_EXTENSION,
            )
