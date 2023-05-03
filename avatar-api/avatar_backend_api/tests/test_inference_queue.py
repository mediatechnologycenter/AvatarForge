import shutil
import unittest
from time import sleep
from unittest import TestCase

from fastapi import UploadFile
from mtc_api_utils.api_types import FirebaseUser

from avatar_backend_api.api_types import AvatarModelRequest, BaseAvatar
from avatar_backend_api.app import io_client
from avatar_backend_api.background_tools.inference_queue import InferenceQueue, InferenceQueueTask
from avatar_backend_api.clients.io_client import IoClient
from avatar_backend_api.clients.mock_avatar_model_client import INFERENCE_DELAY_SECONDS
from avatar_backend_api.models.mock_avatar_model import MockAvatarModel
from avatar_backend_api.tests.test_io_client import TEST_VIDEO_BASE_PATH, TEST_AUDIO_BASE_PATH

TEST_AUDIO_DIR = "/tmp/test-audio"

TEST_UPLOAD_FILE = UploadFile(
    filename="fest-audio-name",
    file=None,
)

TEST_TASK = InferenceQueueTask(
    user=FirebaseUser.example(),
    request=AvatarModelRequest(
        video_id="test-id",
        avatar=BaseAvatar.Jennifer_355_9415.value,
    )
)

TEST_TASK_2 = InferenceQueueTask(
    user=TEST_TASK.user,
    request=AvatarModelRequest(
        video_id=TEST_TASK.request.video_id + "-2",
        avatar=TEST_TASK.request.avatar,
    )
)


class TestInferenceQueue(TestCase):

    def setUp(self) -> None:
        test_io_client = IoClient(user_dir_mode=False, audio_base_path=TEST_AUDIO_BASE_PATH, video_base_path=TEST_VIDEO_BASE_PATH)

        self.inf_queue = InferenceQueue(
            model=MockAvatarModel(io_client=io_client),
            io_client=test_io_client,
            audio_input_dir=TEST_AUDIO_DIR,
            daemon_worker=True,
        )

    def tearDown(self) -> None:
        shutil.rmtree(path=TEST_AUDIO_DIR)

    def test_inference_queue(self):
        self.assertEqual(0, self.inf_queue.queue.qsize())

        self.inf_queue.add_task(TEST_TASK)
        self.assertEqual(1, self.inf_queue.queue.qsize())

        self.inf_queue.add_task(TEST_TASK_2)
        self.assertEqual(2, self.inf_queue.queue.qsize())

        sleep(INFERENCE_DELAY_SECONDS * 3)
        self.assertEqual(0, self.inf_queue.queue.qsize(), msg="Expected worker to process both tasks in the allotted time")


if __name__ == '__main__':
    unittest.main()
