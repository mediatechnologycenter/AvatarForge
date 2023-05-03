import os.path
import shutil
from unittest import TestCase

from fastapi import HTTPException

from avatar_backend_api.api_types import AvatarModel, AvatarModelRequest
from avatar_backend_api.background_tools.model_result_poll_worker import ModelResultPollWorker
from avatar_backend_api.clients.mock_avatar_model_client import MockAvatarModelClient
from avatar_backend_api.tests.test_db_client import TEST_USER, TEST_METADATA, TEST_DB_CLIENT
from avatar_backend_api.tests.test_io_client import TEST_IO_CLIENT_WITH_USER_MODE

TEST_AVATAR_MODEL = AvatarModel.neural_voice

TEST_MODEL_REQUEST = AvatarModelRequest(**TEST_METADATA.json_dict)


class TestResultPollWorker(ModelResultPollWorker):
    """ A test variation of the ModelResultPollWorker which does not call any actions in the background """

    def _worker(self):
        pass


class TestModelResultPollWorker(TestCase):

    def setUp(self) -> None:
        self.poll_worker = TestResultPollWorker(
            io_client=TEST_IO_CLIENT_WITH_USER_MODE,
            db_client=TEST_DB_CLIENT,
            model_clients={TEST_AVATAR_MODEL: MockAvatarModelClient(backend_url="")},
            daemon_worker=True,
        )

        self.poll_worker.db_client.db.drop_tables()
        shutil.rmtree(os.path.join(self.poll_worker.io_client.audio.base_path, "/*"), ignore_errors=True)
        shutil.rmtree(os.path.join(self.poll_worker.io_client.video.base_path, "/*"), ignore_errors=True)

        self.client = self.poll_worker.model_clients[TEST_AVATAR_MODEL]

    def test_no_results(self):
        def fail_on_being_called(video_id: str, avatar_model: AvatarModel):
            self.fail(msg=f"Did not expect any available results for video_id={video_id} and avatar_model={avatar_model.value}")

        self.poll_worker._poll_models(perform_on_available_video=fail_on_being_called)

    def test_result_available(self):
        # Required for retrieving user from video_id
        self.poll_worker.db_client.insert_video(video_metadata=TEST_METADATA, user=TEST_USER)

        # Simulate inference
        self.client.post_audio(audio=(TEST_MODEL_REQUEST.video_id, bytes()), metadata=TEST_MODEL_REQUEST)

        #
        self.poll_worker._retrieve_video_from_model(video_id=TEST_MODEL_REQUEST.video_id, avatar_model=TEST_AVATAR_MODEL)
        self.assertTrue(self.poll_worker.db_client.video_exists(video_id=TEST_MODEL_REQUEST.video_id, user=TEST_USER))

        self.test_no_results()
        self.assertRaises(
            HTTPException,
            lambda: self.poll_worker._retrieve_video_from_model(video_id=TEST_MODEL_REQUEST.video_id, avatar_model=TEST_AVATAR_MODEL),
        )
