# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import logging
from enum import Enum
from http import HTTPStatus
from typing import Union, List

from fastapi import UploadFile, HTTPException, Query, Depends
from fastapi.responses import FileResponse
from mtc_api_utils.api import BaseApi
from mtc_api_utils.api_types import FirebaseUser
from mtc_api_utils.clients.firebase_client import firebase_user_auth

from avatar_backend_api.api_types import ApiRoute, AvatarModelRequest
from avatar_backend_api.background_tools.inference_queue import InferenceQueueTask
from avatar_backend_api.clients.io_client import IoClient, IoClientException
from avatar_backend_api.models.mock_avatar_model import MockAvatarModel
from neural_voice_backend_api.config import NeuralVoiceConfig
from neural_voice_backend_api.neural_voice_inference_queue import NeuralVoiceInferenceQueue
from neural_voice_backend_api.neural_voice_model import NeuralVoiceModel

NeuralVoiceConfig.print_config()

user_auth = firebase_user_auth(config=NeuralVoiceConfig)

io_client = IoClient(audio_base_path=NeuralVoiceConfig.audio_input_dir, video_base_path=NeuralVoiceConfig.video_output_dir, user_dir_mode=False)
neural_voice_model = MockAvatarModel(io_client=io_client) if NeuralVoiceConfig.mockBackend else NeuralVoiceModel(io_client=io_client)

inference_queue = NeuralVoiceInferenceQueue(
    model=neural_voice_model,
    io_client=io_client,
    audio_input_dir=NeuralVoiceConfig.audio_input_dir,
    video_output_dir=NeuralVoiceConfig.video_output_dir,
)

app = BaseApi(is_ready=neural_voice_model.is_ready, config=NeuralVoiceConfig)
tags: List[Union[str, Enum]] = ["Avatar Model"]

logger = logging.Logger("App")


@app.get(
    path=ApiRoute.video_ids.value,
    response_model=List[str],
    dependencies=[Depends(user_auth.with_roles(NeuralVoiceConfig.required_roles))],
    tags=tags,
)
async def list_video_ids() -> List[str]:
    return [path.split("/")[-1].split(".")[0] for path in io_client.video.list_videos()]


@app.get(
    path=ApiRoute.video.value,
    status_code=HTTPStatus.ACCEPTED,
    response_class=FileResponse,
    dependencies=[Depends(user_auth.with_roles(NeuralVoiceConfig.required_roles))],
    tags=tags,
)
async def get_video(video_id: str = Query(alias="videoId")):
    try:
        return io_client.video.read_from_disk(video_id=video_id)

    except IoClientException:
        print(f"")

        if io_client.audio.file_exists(video_id=video_id):
            raise HTTPException(
                status_code=HTTPStatus.PROCESSING,
                detail=f"Audio with {video_id=} is currently being processed",
            )

        else:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"There is no file available for {video_id=}. Either it has not yet been generated or it was removed in the meantime",
            )


@app.post(
    path=ApiRoute.inference.value,
    status_code=HTTPStatus.ACCEPTED,
    response_model=AvatarModelRequest,
    dependencies=[Depends(user_auth.with_roles(NeuralVoiceConfig.required_roles))],
    tags=tags,
)
async def post_audio(
        audio: UploadFile,
        request_metadata: AvatarModelRequest = Depends(),
) -> AvatarModelRequest:
    print(f"Processing {request_metadata.video_id}")

    if not request_metadata.video_id:
        print(f"Creating {request_metadata.video_id=}")
        request_metadata.video_id = audio.filename

    io_client.audio.save_to_disk(audio=audio, video_id=request_metadata.video_id)

    inference_queue.add_task(
        task=InferenceQueueTask(
            request=request_metadata,
        ),
    )

    return request_metadata


@app.delete(
    path=ApiRoute.video.value,
    response_model=str,
    dependencies=[Depends(user_auth)],
    tags=tags,
)
async def delete_video(video_id: str = Query(alias="videoId"), user: FirebaseUser = Depends(user_auth)) -> str:
    try:
        io_client.video.delete_file(video_id=video_id)
    except FileNotFoundError:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"No files with {video_id=} found")

    return f"The files with {video_id=} were removed successfully"
