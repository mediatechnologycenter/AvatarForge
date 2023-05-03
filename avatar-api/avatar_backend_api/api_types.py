from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import uuid4

from mtc_api_utils.api_types import ApiType, FirebaseUser
from pydantic import Field

from avatar_backend_api.config import AvatarConfig


class ApiRoute(Enum):
    value: str

    video = "/api/video"
    list_videos = f"{video}s"
    video_ids = f"{video}/ids"

    inference = "/api/inference"
    avatars = "/api/avatars"

    @staticmethod
    def video_url(backend_url: str, video_id: str) -> str:
        return f"{backend_url}{ApiRoute.video.value}?videoId={video_id}"


class StrEnum(str, Enum):
    value: str


class BaseAvatar(StrEnum):
    Jennifer_355_9415 = "jennifer"
    Arthur_A2226 = "arthur"


Avatar = StrEnum("Avatar", AvatarConfig.available_avatars)


class AvatarModel(Enum):
    value: str
    neural_voice = "neuralVoice"
    motion_gan = "motionGan"


class AvatarModelRequest(ApiType):
    video_id: str = Field(alias="videoId", default_factory=lambda: uuid4().hex)
    avatar: Avatar = Field(example=BaseAvatar.Jennifer_355_9415)


class AvatarRequest(AvatarModelRequest):
    audio_name: str = Field(alias="audioName", example="YourUserName")
    avatar_model: AvatarModel = Field(alias="avatarModel", example=AvatarModel.neural_voice)

    def to_metadata(self, inference_completed: bool = False) -> VideoMetadata:
        return VideoMetadata(**self.json_dict, inference_completed=inference_completed)


class VideoMetadata(AvatarRequest):
    inference_completed: bool = Field(alias="inferenceCompleted", default=False)


@dataclass
class InferenceQueueTask:
    request: AvatarModelRequest
    user: Optional[FirebaseUser] = None
