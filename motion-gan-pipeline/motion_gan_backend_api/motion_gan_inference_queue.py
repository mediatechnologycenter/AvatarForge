# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import shutil

from avatar_backend_api.background_tools.inference_queue import InferenceQueue, InferenceQueueTask
from avatar_backend_api.clients.io_client import VIDEO_EXTENSION

from motion_gan_backend_api.config import MotionGanConfig


class MotionGanInferenceQueue(InferenceQueue):

    def post_processing(self, task: InferenceQueueTask) -> None:
        # Delete input audio
        self.io_client.audio.delete_file(video_id=task.request.video_id)

        # Rename file and return video path
        if not MotionGanConfig.mockBackend:
            self.io_client.video.rename_file(
                filename=f"{task.request.video_id}_to_{task.request.avatar.name}{VIDEO_EXTENSION}",
                new_filename=task.request.video_id + VIDEO_EXTENSION,
            )
