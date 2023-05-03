import os
from time import sleep

from avatar_backend_api.background_tools.inference_queue import AvatarBaseModel, InferenceQueueTask

INFERENCE_DELAY_SECONDS = 1


class MockAvatarModel(AvatarBaseModel):

    def init_model(self):
        pass

    def is_ready(self) -> bool:
        return True

    def inference(self, task: InferenceQueueTask) -> None:
        sleep(INFERENCE_DELAY_SECONDS)
        print(f"Processed audio: {task.request.video_id} with avatar: {task.request.avatar.value}")

        # Create an empty file, representing the output video
        os.makedirs(self.io_client.video.user_dir(user=task.user if self.io_client.user_dir_mode else None), exist_ok=True)

        file_path = self.io_client.video.video_path(video_id=task.request.video_id, user=task.user if self.io_client.user_dir_mode else None)
        with open(file_path, "w") as disk_file:
            disk_file.write("testFile")

        assert os.path.isfile(file_path)
