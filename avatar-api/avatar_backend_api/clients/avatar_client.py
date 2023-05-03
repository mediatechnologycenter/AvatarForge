from typing import List, BinaryIO, Tuple, Union

import requests
from mtc_api_utils.clients.api_client import ApiClient

from avatar_backend_api.api_types import ApiRoute, AvatarModelRequest


class AvatarModelClient(ApiClient):
    def list_video_ids(self) -> List[str]:
        resp = requests.get(
            url=self._backend_url + ApiRoute.video_ids.value,
        )
        resp.raise_for_status()

        return resp.json()

    def get_video(self, video_id: str) -> Tuple[int, bytes]:
        resp = requests.get(
            url=self._backend_url + ApiRoute.video.value,
            params={"videoId": video_id},
        )

        if resp.status_code >= 300:
            resp.raise_for_status()

        return resp.status_code, resp.content

    def post_audio(self, audio: Union[BinaryIO, Tuple[str, bytes]], metadata: AvatarModelRequest) -> AvatarModelRequest:
        resp = requests.post(
            url=self._backend_url + ApiRoute.inference.value,
            params=metadata.json_dict,
            files={"audio": audio},
        )

        if resp.status_code >= 300:
            print(f"resp.reason={resp.reason}")
            print(f"resp.text={resp.text}")
            print(f"resp.raw={resp.raw}")

        resp.raise_for_status()

        return metadata

    def delete_video(self, video_id: str) -> str:
        resp = requests.delete(ApiRoute.video_url(backend_url=self._backend_url, video_id=video_id))
        resp.raise_for_status()

        return resp.text
