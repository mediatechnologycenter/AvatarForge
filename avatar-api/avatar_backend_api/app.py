from http import HTTPStatus
from typing import List, Dict
from uuid import uuid4

import uvicorn
from fastapi import UploadFile, Depends, Query, HTTPException
from mtc_api_utils.api import BaseApi, RequireReadinessDependency
from mtc_api_utils.api_types import FirebaseUser
from mtc_api_utils.clients.firebase_client import firebase_user_auth
from starlette.responses import FileResponse

from avatar_backend_api.api_types import ApiRoute, VideoMetadata, AvatarRequest, AvatarModel
from avatar_backend_api.background_tools.model_result_poll_worker import ModelResultPollWorker
from avatar_backend_api.clients.avatar_client import AvatarModelClient
from avatar_backend_api.clients.db_client import AvatarDbClient
from avatar_backend_api.clients.io_client import IoClient, IoClientException
from avatar_backend_api.clients.mock_avatar_model_client import MockAvatarModelClient
from avatar_backend_api.config import AvatarConfig

AvatarConfig.print_config()

user_auth = firebase_user_auth(config=AvatarConfig)

db_client = AvatarDbClient()
io_client = IoClient(user_dir_mode=True)

model_clients: Dict[AvatarModel, AvatarModelClient] = {
    AvatarModel.neural_voice:
        AvatarModelClient(backend_url=AvatarConfig.neural_voice_backend_url)
        if not AvatarConfig.mockBackend
        else MockAvatarModelClient(backend_url=""),
    AvatarModel.motion_gan:
        AvatarModelClient(backend_url=AvatarConfig.motion_gan_backend_url)
        if not AvatarConfig.mockBackend
        else MockAvatarModelClient(backend_url=""),
}

model_poll_worker = ModelResultPollWorker(
    io_client=io_client,
    db_client=db_client,
    model_clients=model_clients,
)


def backend_is_ready() -> bool:
    # Only turn ready if both models are ready
    for model, client in model_clients.items():
        if model.value in AvatarConfig.avatar_models:
            _, is_ready = client.get_readiness()
            if not is_ready:
                return False

    return True


app = BaseApi(is_ready=backend_is_ready, config=AvatarConfig, global_readiness_middleware_enabled=False)


@app.get(
    path=ApiRoute.video_ids.value,
    response_model=List[str],
)
async def list_video_ids(user: FirebaseUser = Depends(user_auth)) -> List[str]:
    return [path.split("/")[-1].split(".")[0] for path in io_client.video.list_videos(user=user)]


@app.get(
    path=ApiRoute.list_videos.value,
    response_model=List[VideoMetadata],
)
async def list_videos(user: FirebaseUser = Depends(user_auth)) -> List[VideoMetadata]:
    return db_client.list_videos(user=user)


@app.get(
    path=ApiRoute.video.value,
    response_class=FileResponse,
)
async def get_video(video_id: str = Query(alias="videoId")):  # user: FirebaseUser = Depends(user_auth)
    # The following line represents a workaround for the fact that the frontend video library can't send an authentication header with the video request.
    # If the library should be fixed in the future, the user should be retrieved using the user_auth dependency as follows: [ user: FirebaseUser = Depends(user_auth) ]
    user = FirebaseUser(email=db_client.get_user_from_id(video_id=video_id), roles=[])

    try:
        return io_client.video.read_from_disk(video_id=video_id, user=user)
    except IoClientException:
        if db_client.video_exists(video_id=video_id, user=user):
            raise HTTPException(
                status_code=HTTPStatus.PROCESSING,
                detail=f"Audio with {video_id=} is currently being processed",
            )

        else:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"A request with video_id={video_id} has not been found")


@app.post(
    path=ApiRoute.inference.value,
    status_code=HTTPStatus.ACCEPTED,
    response_model=AvatarRequest,
    dependencies=[Depends(RequireReadinessDependency(base_api=app))],

)
async def inference(
        audio: UploadFile,
        request_metadata: AvatarRequest = Depends(),
        user: FirebaseUser = Depends(user_auth),
) -> AvatarRequest:
    print(f"Routing request to model={request_metadata.avatar_model.value} for audio_name={audio.filename}")

    if not request_metadata.video_id:
        request_metadata.video_id = uuid4().hex

    if "." not in audio.filename:
        audio.filename += ".wav"

    metadata = request_metadata.to_metadata()
    filename = f"{request_metadata.video_id}.{audio.filename.split('.')[-1]}"

    model_clients[request_metadata.avatar_model].post_audio(audio=(filename, audio.file.read()), metadata=metadata)
    db_client.insert_video(video_metadata=metadata, user=user)

    return metadata


@app.delete(
    path=ApiRoute.video.value,
    response_model=str,
)
async def delete_video(video_id: str = Query(alias="videoId"), user: FirebaseUser = Depends(user_auth)) -> str:
    db_client.delete_video(video_id=video_id, user=user)

    try:
        io_client.video.delete_file(video_id=video_id, user=user)

    except IoClientException:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail=f"Unable to find video with video_id={video_id}")

    return f"Successfully removed video with video_id={video_id}"


@app.get(
    path=ApiRoute.avatars.value,
    response_model=Dict[str, str],
    tags=["Avatars"],
    dependencies=[Depends(user_auth)],
)
async def available_avatars() -> Dict[str, str]:
    return AvatarConfig.available_avatars


if __name__ == '__main__':
    uvicorn.run(app=app, port=5000)
