from http import HTTPStatus
from time import sleep
from unittest import TestCase

from fastapi import HTTPException
from httpx import HTTPStatusError

from avatar_backend_api.api_types import AvatarModel, AvatarRequest, BaseAvatar
from avatar_backend_api.clients.mock_avatar_model_client import INFERENCE_DELAY_SECONDS
from avatar_backend_api.config import AvatarConfig
from avatar_backend_api.tests.integration_test_client import AvatarModelTestClient
from avatar_backend_api.tests.test_io_client import TEST_FILE_PATH

TEST_INFERENCE_REQUEST = AvatarRequest(
    video_id="test-id",
    audio_name="test-video-name",
    avatar=BaseAvatar.Jennifer_355_9415.value,
    avatar_model=AvatarModel.neural_voice,
)

TEST_INFERENCE_REQUEST_2 = AvatarRequest(
    video_id=TEST_INFERENCE_REQUEST.video_id + "-2",
    audio_name=TEST_INFERENCE_REQUEST.audio_name + "-2",
    avatar=TEST_INFERENCE_REQUEST.avatar,
    avatar_model=TEST_INFERENCE_REQUEST.avatar_model,
)


class TestIntegration(TestCase):
    client = AvatarModelTestClient()

    @classmethod
    def setUpClass(cls) -> None:
        cls.client.wait_for_service_readiness()

    def setUp(self) -> None:
        try:
            self.client.delete_video(video_id=TEST_INFERENCE_REQUEST.video_id)
        except (HTTPException, HTTPStatusError):
            pass

        try:
            self.client.delete_video(video_id=TEST_INFERENCE_REQUEST_2.video_id)
        except (HTTPException, HTTPStatusError):
            pass

    def tearDown(self) -> None:
        self.setUp()

    def test_api_test(self):
        self.assertRaises((HTTPException, HTTPStatusError), lambda: self.client.get_video(video_id=TEST_INFERENCE_REQUEST.video_id))

        with open(TEST_FILE_PATH, "rb") as test_audio:
            created_video = self.client.post_audio(audio=test_audio, metadata=TEST_INFERENCE_REQUEST)
            self.assertEqual(TEST_INFERENCE_REQUEST.video_id, created_video.video_id)

        self.assertIn(TEST_INFERENCE_REQUEST.video_id, [vid.video_id for vid in self.client.list_videos()])

        # Wait for result to be available
        while True:
            status_code, file_bytes = self.client.get_video(created_video.video_id)
            self.assertIn(status_code, [HTTPStatus.OK, HTTPStatus.PROCESSING])

            if status_code == HTTPStatus.OK:
                break

        if not AvatarConfig.mockBackend:
            self.assertGreater(len(file_bytes), 0)

        self.client.delete_video(created_video.video_id)
        self.assertNotIn(TEST_INFERENCE_REQUEST.video_id, [vid.video_id for vid in self.client.list_videos()])

    def test_worker_test(self):
        if not AvatarConfig.mockBackend:
            self.skipTest("This test should only be performed if the model is mocked, otherwise it takes way too long and the delay is incorrect")

        with open(TEST_FILE_PATH, "rb") as test_audio:
            self.client.post_audio(audio=test_audio, metadata=TEST_INFERENCE_REQUEST)
            self.client.post_audio(audio=test_audio, metadata=TEST_INFERENCE_REQUEST_2)

        sleep(INFERENCE_DELAY_SECONDS * 3)
        video_ids = self.client.list_video_ids()

        self.assertIn(TEST_INFERENCE_REQUEST.video_id, video_ids)
        self.assertIn(TEST_INFERENCE_REQUEST.video_id, video_ids)
