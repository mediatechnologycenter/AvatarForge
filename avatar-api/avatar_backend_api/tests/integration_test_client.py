from typing import List, Tuple, Union, BinaryIO

from fastapi.testclient import TestClient

from avatar_backend_api.api_types import VideoMetadata, ApiRoute, AvatarRequest
from avatar_backend_api.app import app
from avatar_backend_api.clients.avatar_client import AvatarModelClient


class AvatarModelTestClient(AvatarModelClient):
    def __init__(self, backend_url="", http_client=TestClient(app=app)):
        super().__init__(backend_url=backend_url, http_client=http_client)

    def list_video_ids(self) -> List[str]:
        resp = self.http_client.get(url=ApiRoute.video_ids.value)
        resp.raise_for_status()

        return resp.json()

    def list_videos(self) -> List[VideoMetadata]:
        resp = self.http_client.get(url=ApiRoute.list_videos.value)
        resp.raise_for_status()

        return [VideoMetadata.parse_obj(vid) for vid in resp.json()]

    def get_video(self, video_id: str) -> Tuple[int, bytes]:
        """ Returns the response code as well as the bytes representing the retrieved file"""
        resp = self.http_client.get(ApiRoute.video_url(backend_url="", video_id=video_id))

        if resp.status_code >= 300:
            resp.raise_for_status()

        return resp.status_code, resp.content

    def post_audio(self, audio: Union[BinaryIO, Tuple[str, bytes]], metadata: AvatarRequest) -> AvatarRequest:
        """ Overrides the AvatarBaseModel with AvatarRequest in order to send extended schema to backend for testing """
        resp = self.http_client.post(
            url=ApiRoute.inference.value,
            params=metadata.json_dict,
            files={"audio": audio}
        )

        if resp.status_code >= 300:
            print(resp.text)
            resp.raise_for_status()

        return AvatarRequest.parse_obj(resp.json())

    def delete_video(self, video_id: str) -> str:
        resp = self.http_client.delete(ApiRoute.video_url(backend_url="",video_id=video_id))
        resp.raise_for_status()

        return resp.text
