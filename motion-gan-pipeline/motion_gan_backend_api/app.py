# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import logging
from http import HTTPStatus
from typing import List, Dict

from fastapi import UploadFile, Depends, Query, HTTPException
from fastapi.responses import FileResponse
from mtc_api_utils.api import BaseApi
from mtc_api_utils.clients.firebase_client import firebase_user_auth

from avatar_backend_api.api_types import ApiRoute, InferenceQueueTask, AvatarModelRequest
from avatar_backend_api.clients.io_client import IoClient, IoClientException
from avatar_backend_api.models.avatar_base_model import AvatarBaseModel
from avatar_backend_api.models.mock_avatar_model import MockAvatarModel
from motion_gan_backend_api.config import MotionGanConfig
from motion_gan_backend_api.motion_gan_inference_queue import MotionGanInferenceQueue
from motion_gan_backend_api.motion_gan_model import MotionGANModel

MotionGanConfig.print_config()

user_auth = firebase_user_auth(config=MotionGanConfig)

io_client = IoClient(audio_base_path=MotionGanConfig.audio_input_dir, video_base_path=MotionGanConfig.video_output_dir, user_dir_mode=False)
motion_gan_model: AvatarBaseModel = MockAvatarModel(io_client=io_client) if MotionGanConfig.mockBackend else MotionGANModel(io_client=io_client)

inference_queue = MotionGanInferenceQueue(
    model=motion_gan_model,
    io_client=io_client,
    audio_input_dir=MotionGanConfig.audio_input_dir,
    video_output_dir=MotionGanConfig.video_output_dir,
)

app = BaseApi(is_ready=motion_gan_model.is_ready, config=MotionGanConfig)
tags = ["Avatar Model"]

logger = logging.Logger("App")


@app.get(
    path=ApiRoute.video_ids.value,
    response_model=List[str],
    tags=tags,
    dependencies=[Depends(user_auth)],
)
async def list_video_ids() -> List[str]:
    return [path.split("/")[-1].split(".")[0] for path in io_client.video.list_videos()]


@app.get(
    path=ApiRoute.video.value,
    status_code=HTTPStatus.ACCEPTED,
    response_class=FileResponse,
    tags=tags,
    dependencies=[Depends(user_auth)],
)
async def get_video(video_id: str = Query(alias="videoId")):  # -> FileResponse: # For some reason does not seem to work with python 3.7
    try:
        return io_client.video.read_from_disk(video_id=video_id)

    except IoClientException:
        if io_client.audio.file_exists(video_id=video_id):
            raise HTTPException(
                status_code=HTTPStatus.PROCESSING,
                detail=f"Audio with video_id={video_id} is currently being processed",
            )

        else:
            raise HTTPException(
                status_code=HTTPStatus.NOT_FOUND,
                detail=f"There is no file available for video_id={video_id}. Either it has not yet been generated or it was removed in the meantime",
            )


@app.post(
    path=ApiRoute.inference.value,
    status_code=HTTPStatus.ACCEPTED,
    response_model=AvatarModelRequest,
    tags=tags,
    dependencies=[Depends(user_auth)],
)
async def model_inference(audio: UploadFile, request_metadata: AvatarModelRequest = Depends()) -> AvatarModelRequest:
    print(f"Processing request for video_id={request_metadata.video_id}")

    if not request_metadata.video_id:
        print(f"Creating video_id={request_metadata.video_id}")
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
    tags=tags,
    dependencies=[Depends(user_auth)],
)
async def delete_video(video_id: str = Query(alias="videoId")) -> str:
    try:
        io_client.video.delete_file(video_id=video_id)
    except FileNotFoundError:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"No files with video_id={video_id} found")

    return f"The files with video_id={video_id} were removed successfully"


@app.get(
    path=ApiRoute.avatars.value,
    response_model=Dict[str, str],
    tags=["Avatars"],
    dependencies=[Depends(user_auth)],
)
async def available_avatars() -> Dict[str, str]:
    return MotionGANModel.available_avatars()
