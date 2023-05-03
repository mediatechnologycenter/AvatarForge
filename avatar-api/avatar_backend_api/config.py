from __future__ import annotations

import os
from typing import Dict, List

from mtc_api_utils.config import Config


def _avatar_short_name(avatar_name: str) -> str:
    return avatar_name.split("_")[0].split("-")[0].split(".")[0].lower()


class AvatarConfig(Config):
    # API configs
    neural_voice_backend_url: str = Config.parse_env_var("NEURAL_VOICE_BACKEND_URL", default="http://neural-voice-model:5000")
    motion_gan_backend_url: str = Config.parse_env_var("MOTION_GAN_BACKEND_URL", default="http://motion-gan-model:5000")

    # Debug configs
    mockBackend: bool = Config.parse_env_var("MOCK_BACKEND", default="False", convert_type=bool)

    # IO configs
    avatar_models: List[str] = Config.parse_env_var("AVATAR_MODELS", default="neuralVoice,motionGan", convert_type=list)
    available_avatars: Dict[str, str] = {
        avatar: _avatar_short_name(avatar)
        for avatar in list(Config.parse_env_var("AVAILABLE_AVATARS", default="Jennifer_355_9415,Arthur_A2226", convert_type=list))
    }

    db_filepath: str = Config.parse_env_var("DB_FILEPATH", default="/tmp/tinyDB/neuralVoices.json")

    data_base_dir: str = Config.parse_env_var("DATA_BASE_DIR", default="/tmp/avatar")
    audio_input_dir: str = os.path.join(data_base_dir, "input_data", "audio")
    video_output_dir: str = os.path.join(data_base_dir, "output_data", "videos")

    @staticmethod
    def avatar_short_name(avatar_name: str) -> str:
        return _avatar_short_name(avatar_name=avatar_name)
