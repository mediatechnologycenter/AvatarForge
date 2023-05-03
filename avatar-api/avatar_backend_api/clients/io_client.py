import os.path
from glob import glob
from http import HTTPStatus
from typing import BinaryIO, List, Optional, Union

from fastapi import UploadFile, HTTPException
from mtc_api_utils.api_types import FirebaseUser
from starlette.responses import FileResponse

from avatar_backend_api.config import AvatarConfig

VIDEO_EXTENSION = ".mp4"


class IoClientException(HTTPException):
    pass


class BaseIOClient:
    def __init__(self, base_path: str, user_dir_mode: bool):
        self.base_path = base_path
        self.user_dir_mode = user_dir_mode

    def user_dir(self, user: Optional[FirebaseUser]) -> str:
        if self.user_dir_mode:
            if user is None:
                raise ValueError("In user_dir_mode, a user is expected for IoClient methods")
            else:
                return os.path.join(self.base_path, user.email)

        if not self.user_dir_mode:
            if user is not None:
                raise ValueError("User is only expected in user_dir_mode. Either remove it or enable user_dir_mode in the IoClient constructor")
            else:
                return self.base_path


class AudioIoClient(BaseIOClient):
    def matching_paths(self, video_id: str, user: Optional[FirebaseUser] = None) -> List[str]:
        matching_paths = sorted(glob(os.path.join(self.user_dir(user=user), video_id, video_id) + ".*"))

        if len(matching_paths) == 0:
            raise IoClientException(status_code=HTTPStatus.NOT_FOUND, detail=f"Unable to find file with video_id={video_id}")

        return matching_paths

    def audio_path(self, video_id: str, file_ending: str, user: Optional[FirebaseUser] = None) -> str:
        return os.path.join(self.user_dir(user=user), video_id, f"{video_id}.{file_ending}")

    def file_exists(self, video_id: str, user: Optional[FirebaseUser] = None) -> bool:
        try:
            paths = self.matching_paths(video_id=video_id, user=user)
        except IoClientException:
            return False

        return len(paths) > 0

    def save_to_disk(self, audio: UploadFile, video_id: str, user: Optional[FirebaseUser] = None) -> str:
        file_ending = audio.filename.split(".")[-1]

        path = self.audio_path(video_id=video_id, file_ending=file_ending, user=user)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as disk_file:
            file_bytes = audio.file.read()
            disk_file.write(file_bytes)

        return path

    def read_from_disk(self, video_id: str, user: Optional[FirebaseUser] = None) -> BinaryIO:
        try:
            return open(self.matching_paths(video_id=video_id, user=user)[-1], "rb")
        except FileNotFoundError:
            raise IoClientException(status_code=HTTPStatus.NOT_FOUND, detail=f"Unable to find video with video_id={video_id}")

    def delete_file(self, video_id: str, user: Optional[FirebaseUser] = None) -> None:
        for path in self.matching_paths(video_id=video_id, user=user):
            try:
                os.remove(path)
            except FileNotFoundError:
                raise IoClientException(status_code=HTTPStatus.NOT_FOUND, detail=f"Unable to find video with video_id={video_id}")


class VideoIoClient(BaseIOClient):
    def list_videos(self, user: Optional[FirebaseUser] = None) -> List[str]:
        return sorted(glob(os.path.join(self.user_dir(user=user), f"*{VIDEO_EXTENSION}")))

    def video_path(self, video_id: str, user: Optional[FirebaseUser] = None) -> str:
        return os.path.join(self.user_dir(user=user), video_id + VIDEO_EXTENSION)

    def file_exists(self, video_id: str, user: Optional[FirebaseUser] = None) -> bool:
        return os.path.isfile(self.video_path(video_id=video_id, user=user))

    def save_to_disk(self, video_bytes: Union[bytes, BinaryIO], video_id: str, user: Optional[FirebaseUser] = None) -> str:
        try:
            video_bytes = video_bytes.read()
        except AttributeError:
            pass

        path = self.video_path(video_id=video_id, user=user)

        os.makedirs(self.user_dir(user=user), exist_ok=True)
        with open(path, "wb") as disk_file:
            disk_file.write(video_bytes)

        return path

    def read_from_disk(self, video_id: str, user: Optional[FirebaseUser] = None) -> FileResponse:
        file_path = self.video_path(video_id=video_id, user=user)

        if os.path.isfile(file_path):
            return FileResponse(
                path=file_path,
            )

        else:
            raise IoClientException(status_code=HTTPStatus.NOT_FOUND, detail=f"Unable to find video with video_id={video_id}")

    def rename_file(self, filename: str, new_filename: str, user: Optional[FirebaseUser] = None) -> str:
        input_path = os.path.join(self.user_dir(user=user), filename)
        output_path = os.path.join(self.user_dir(user=user), new_filename)

        try:
            os.rename(
                src=input_path,
                dst=output_path,
            )
        except FileNotFoundError:
            raise IoClientException(status_code=HTTPStatus.NOT_FOUND, detail=f"Unable to find video with filename={filename}")

        return output_path

    def delete_file(self, video_id: str, user: Optional[FirebaseUser] = None) -> str:
        try:
            path = self.video_path(video_id=video_id, user=user)
            os.remove(path)
            return path

        except FileNotFoundError:
            raise IoClientException(status_code=HTTPStatus.NOT_FOUND, detail=f"Unable to find video with video_id={video_id}")


class IoClient:
    """ Manages file input & output for Avatar audio & video files required by the Avatar project.
    Can be used in two modes: If a FirebaseUser is passed to a method, the files will be read from and written to a subdirectory named after the user.email.
    If the user is omitted,
    """

    def __init__(
            self,
            user_dir_mode: bool,
            audio_base_path: str = AvatarConfig.audio_input_dir,
            video_base_path: str = AvatarConfig.video_output_dir,
    ):
        self.user_dir_mode = user_dir_mode
        self.audio = AudioIoClient(base_path=audio_base_path, user_dir_mode=user_dir_mode)
        self.video = VideoIoClient(base_path=video_base_path, user_dir_mode=user_dir_mode)
