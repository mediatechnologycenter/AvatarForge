import os
from http import HTTPStatus
from typing import List

from fastapi import HTTPException
from mtc_api_utils.api_types import FirebaseUser
from tinydb import TinyDB, where
from tinydb.table import Table

from avatar_backend_api.api_types import VideoMetadata
from avatar_backend_api.config import AvatarConfig


class AvatarDbException(HTTPException):
    pass


class AvatarDbClient:
    db_filepath: str
    db: TinyDB

    def __init__(self, db_filepath: str = AvatarConfig.db_filepath):
        os.makedirs(
            name=os.path.dirname(db_filepath),
            exist_ok=True,
        )

        self.db_filepath = db_filepath
        self.db = TinyDB(db_filepath)

    def _user_table(self, user: FirebaseUser) -> Table:
        return self.db.table(name=user.email)

    def list_videos(self, user: FirebaseUser) -> List[VideoMetadata]:
        return [VideoMetadata.parse_obj(doc) for doc in self._user_table(user=user).all()]

    def video_exists(self, video_id: str, user: FirebaseUser) -> bool:
        return self._user_table(user=user).count(self.video_query(video_id=video_id)) == 1

    def get_video(self, video_id: str, user: FirebaseUser) -> VideoMetadata:
        result = self._user_table(user=user).get(self.video_query(video_id=video_id))

        if result is None:
            raise AvatarDbException(status_code=HTTPStatus.NOT_FOUND, detail=f"Video with id: {video_id} does not exist")
        else:
            return VideoMetadata.parse_obj(result)

    def insert_video(self, video_metadata: VideoMetadata, user: FirebaseUser) -> VideoMetadata:
        if self._user_table(user=user).count(cond=self.video_query(video_id=video_metadata.video_id)) != 0:
            raise AvatarDbException(HTTPStatus.CONFLICT, detail=f"An inference request with id {video_metadata.video_id} already exists")

        self._user_table(user=user).insert(document=video_metadata.json_dict)
        print(f"Inserting new video_metadata for user={user} with video_id={video_metadata.video_id}")

        return video_metadata

    def upsert_video(self, video_metadata: VideoMetadata, user: FirebaseUser) -> VideoMetadata:
        # Update doc if exists, else insert it
        self._user_table(user=user).upsert(
            document=video_metadata.json_dict,
            cond=self.video_query(video_id=video_metadata.video_id),
        )
        print(f"Updating video_metadata for user={user} with video_id={video_metadata.video_id}")

        return video_metadata

    def delete_video(self, video_id: str, user: FirebaseUser) -> VideoMetadata:
        video_metadata = self.get_video(video_id=video_id, user=user)
        self._user_table(user=user).remove(self.video_query(video_id=video_id))
        print(f"Removing video_metadata for user={user} with video_id={video_metadata.video_id}")

        return video_metadata

    def get_user_from_id(self, video_id) -> str:
        for user_email in self.db.tables():
            if self.db.table(name=user_email).contains(self.video_query(video_id=video_id)):
                return user_email

        raise AvatarDbException(status_code=HTTPStatus.NOT_FOUND, detail=f"Video with id: {video_id} does not exist in db. Existing users & video_ids: {[(email, self._user_table(user=FirebaseUser(email=email, roles=[])).all()) for email in self.db.tables()]}")

    @staticmethod
    def video_query(video_id: str):
        return where("videoId") == video_id
