# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from http import HTTPStatus
from typing import List, BinaryIO, Tuple

from avatar_backend_api.api_types import ApiRoute, AvatarModelRequest
from mtc_api_utils.api import BaseApi
from mtc_api_utils.clients.api_client import ApiClient
from starlette.testclient import TestClient

from avatar_backend_api.config import AvatarConfig


class AvatarIntegrationTestClient(ApiClient):

    def __init__(self, test_app: BaseApi):
        super().__init__(backend_url="", http_client=TestClient(app=test_app))

    def list_video_ids(self) -> List[str]:
        resp = self.http_client.get(
            url=ApiRoute.video_ids.value,
        )
        resp.raise_for_status()

        return resp.json()

    def get_video(self, video_id: str) -> Tuple[int, bytes]:
        resp = self.http_client.get(
            url=ApiRoute.video.value,
            params={"videoId": video_id},
        )

        if resp.status_code >= 300:
            resp.raise_for_status()

        return resp.status_code, resp.content

    def post_audio(self, audio: BinaryIO, metadata: AvatarModelRequest) -> AvatarModelRequest:
        resp = self.http_client.post(
            url=ApiRoute.inference.value,
            params=metadata.json_dict,
            files={"audio": audio},
        )

        if resp.status_code >= 300:
            print(f"{resp.reason=}")
            print(f"{resp.text=}")
            print(f"{resp.raw=}")

        resp.raise_for_status()
        assert resp.status_code == HTTPStatus.ACCEPTED

        return AvatarModelRequest.parse_obj(resp.json())

    def delete_video(self, video_id: str) -> str:
        resp = self.http_client.delete(ApiRoute.video_url(backend_url="", video_id=video_id))
        resp.raise_for_status()

        return resp.text
