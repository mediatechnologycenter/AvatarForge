import logging
from http import HTTPStatus
from time import sleep
from typing import Dict, Union, BinaryIO, Tuple, List, Optional

import requests
from fastapi import HTTPException
from mtc_api_utils.api_types import ApiStatus

from avatar_backend_api.api_types import AvatarModelRequest
from avatar_backend_api.clients.avatar_client import AvatarModelClient

log = logging.Logger("MockNeuralVoiceModel-Logger")

INFERENCE_DELAY_SECONDS = 1

mock_db: Dict[str, AvatarModelRequest] = {}


class MockAvatarModelClient(AvatarModelClient):

    def get_liveness(self) -> Tuple[Optional[requests.Response], bool]:
        return requests.Response(), True

    def get_readiness(self) -> Tuple[Optional[requests.Response], bool]:
        return requests.Response(), True

    def get_status(self) -> Tuple[Optional[requests.Response], ApiStatus]:
        return requests.Response(), ApiStatus(readiness=self.get_readiness()[1], gpu_supported=True, gpu_enabled=True)

    def list_video_ids(self) -> List[str]:
        return [metadata.video_id for metadata in mock_db.values()]

    def get_video(self, video_id: str) -> Tuple[int, bytes]:
        return HTTPStatus.OK, bytes()

    def post_audio(self, audio: Union[BinaryIO, Tuple[str, bytes]], metadata: AvatarModelRequest) -> AvatarModelRequest:
        sleep(INFERENCE_DELAY_SECONDS)

        mock_db[metadata.video_id] = metadata
        print(f"Processed audio: {metadata.video_id}")

        return metadata

    def delete_video(self, video_id: str) -> str:
        try:
            del mock_db[video_id]
        except KeyError:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Video with video_id={video_id} was not found")

        return f"Successfully deleted video with video_id={video_id}"
