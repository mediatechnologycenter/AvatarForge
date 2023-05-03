# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os.path

from avatar_backend_api.config import AvatarConfig
from mtc_api_utils.config import Config


class NeuralVoiceConfig(AvatarConfig):
    mockBackend: bool = Config.parse_env_var("MOCK_BACKEND", default="False", convert_type=bool)

    # Data configs
    db_filepath: str = Config.parse_env_var("DB_FILEPATH", default="/tmp/tinyDB/neuralVoices.json")

    neural_code_base_dir: str = "/app/neural-code"  # Location of the neural-code dir in the execution environment

    data_base_dir: str = Config.parse_env_var("DATA_BASE_DIR", default="/tmp/neuralVoice")
    audio_input_dir: str = os.path.join(data_base_dir, "input_data", "audio")
    video_input_dir: str = os.path.join(data_base_dir, "input_data", "video")

    output_data_path: str = os.path.join(data_base_dir, "output_data")
    features_dir: str = os.path.join(output_data_path, "features")
    video_output_dir: str = os.path.join(output_data_path, "videos")
    checkpoints_dir: str = os.path.join(output_data_path, "checkpoints")
    mappings_dir = os.path.join(output_data_path, "mappings")
