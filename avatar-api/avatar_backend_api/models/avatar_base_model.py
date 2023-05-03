from abc import ABC, abstractmethod
from typing import List

from mtc_api_utils.base_model import MLBaseModel

from avatar_backend_api.api_types import InferenceQueueTask
from avatar_backend_api.clients.io_client import IoClient


class AvatarBaseModel(MLBaseModel, ABC):

    def __init__(self, io_client: IoClient):
        self.io_client = io_client
        super().__init__()

    @abstractmethod
    def inference(self, task: InferenceQueueTask) -> None:
        raise Exception("Not implemented")

    @staticmethod
    @abstractmethod
    def available_avatars() -> List[str]:
        raise Exception("Not implemented")
