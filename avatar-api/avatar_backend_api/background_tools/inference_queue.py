import os
from queue import SimpleQueue
from threading import Thread
from time import sleep

from avatar_backend_api.api_types import InferenceQueueTask
from avatar_backend_api.clients.io_client import IoClient
from avatar_backend_api.config import AvatarConfig
from avatar_backend_api.models.avatar_base_model import AvatarBaseModel


class InferenceQueue:
    def __init__(
            self,
            model: AvatarBaseModel,
            io_client: IoClient,
            audio_input_dir: str = AvatarConfig.audio_input_dir,
            video_output_dir: str = AvatarConfig.video_output_dir,
            daemon_worker: bool = True,
    ):
        self.queue: SimpleQueue[InferenceQueueTask] = SimpleQueue()

        self.model = model
        self.io_client = io_client

        self.audio_input_dir = audio_input_dir
        self.video_output_dir = video_output_dir
        self._worker_thread = Thread(target=self._worker, daemon=daemon_worker)

        os.makedirs(self.audio_input_dir, exist_ok=True)
        os.makedirs(self.video_output_dir, exist_ok=True)

        self._worker_thread.start()

    def post_processing(self, task: InferenceQueueTask) -> None:
        pass

    def add_task(self, task: InferenceQueueTask) -> None:
        self.queue.put(task)

    def _worker(self):
        while True:
            if self.model.is_ready() and not self.queue.empty():
                task = self.queue.get()
                self.model.inference(task=task)
                self.post_processing(task=task)
            else:
                sleep(1)
