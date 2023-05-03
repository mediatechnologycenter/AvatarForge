from threading import Thread
from time import sleep
from typing import Dict, Callable

from mtc_api_utils.api_types import FirebaseUser

from avatar_backend_api.api_types import AvatarModel
from avatar_backend_api.clients.avatar_client import AvatarModelClient
from avatar_backend_api.clients.db_client import AvatarDbClient, AvatarDbException
from avatar_backend_api.clients.io_client import IoClient


class ModelResultPollWorker:

    def __init__(
            self,
            io_client: IoClient,
            db_client: AvatarDbClient,
            model_clients: Dict[AvatarModel, AvatarModelClient],
            daemon_worker: bool = True,
    ):
        self.io_client = io_client
        self.model_clients = model_clients
        self.db_client = db_client

        self._worker_thread = Thread(target=self._worker, daemon=daemon_worker)
        self._worker_thread.start()

    def _worker(self):
        """
        Continuously polls processed videos from model backends, retrieves them and performs cleanup.

        Methods are separated mostly for testing purposes
        """
        while True:
            self._poll_models(perform_on_available_video=self._retrieve_video_from_model)
            sleep(5)

    def _poll_models(self, perform_on_available_video: Callable[[str, AvatarModel], None]):
        """ GET available videos from Model, then GET video & store video file + metadata in backend API"""

        for model, client in self.model_clients.items():
            if client.get_readiness()[1]:
                video_ids = client.list_video_ids()
                for video_id in video_ids:
                    print(f"Retrieving the following videos from model={model}: {video_ids}")
                    perform_on_available_video(video_id, model)

    def _retrieve_video_from_model(self, video_id: str, avatar_model: AvatarModel) -> None:
        status, video_bytes = self.model_clients[avatar_model].get_video(video_id=video_id)

        try:
            user = FirebaseUser(
                email=self.db_client.get_user_from_id(video_id=video_id),
                roles=[],
            )

        except AvatarDbException as e:
            print(e.detail)
            return

        video_metadata = self.db_client.get_video(video_id=video_id, user=user)
        video_metadata.inference_completed = True

        self.io_client.video.save_to_disk(video_bytes=video_bytes, video_id=video_id, user=user)
        self.db_client.upsert_video(video_metadata=video_metadata, user=user)

        print(f"Deleting video from model")
        self.model_clients[avatar_model].delete_video(video_id=video_id)
