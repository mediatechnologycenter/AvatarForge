import os.path
import shutil
from enum import Enum
from unittest import TestCase

from avatar_backend_api.clients.io_client import IoClient, IoClientException
from avatar_backend_api.tests.test_db_client import TEST_USER
from fastapi import UploadFile
from mtc_api_utils.api_types import FirebaseUser


class TestFile(Enum):
    value: str
    test_mp3 = "test.mp3"
    ping_wav = "ping.wav"
    voice_wav = "voice_test.wav"


TEST_FILENAME = TestFile.voice_wav.value

TEST_BASE_PATH = "/tmp/test"
TEST_AUDIO_BASE_PATH = os.path.join(TEST_BASE_PATH, "audio")
TEST_VIDEO_BASE_PATH = os.path.join(TEST_BASE_PATH, "video")

TEST_ID = "test-id"
TEST_ID_2 = TEST_ID + "-2"

TEST_USER_2 = FirebaseUser(
    email=TEST_USER.email + "-2",
    roles=TEST_USER.roles,
)

# test_dir represents an absolute path to the test directory, independent of where the test was started from. Use this as a base dir for any path operations.
test_dir = os.path.dirname(os.path.realpath(__file__))
TEST_FILE_PATH = os.path.join(test_dir, TEST_FILENAME)

TEST_IO_CLIENT = IoClient(user_dir_mode=False, audio_base_path=TEST_AUDIO_BASE_PATH, video_base_path=TEST_VIDEO_BASE_PATH)
TEST_IO_CLIENT_WITH_USER_MODE = IoClient(user_dir_mode=True, audio_base_path=TEST_AUDIO_BASE_PATH, video_base_path=TEST_VIDEO_BASE_PATH)


class TestIoClient(TestCase):
    def setUp(self) -> None:
        shutil.rmtree(TEST_BASE_PATH, ignore_errors=True)

        os.makedirs(TEST_AUDIO_BASE_PATH, exist_ok=True)
        os.makedirs(TEST_VIDEO_BASE_PATH, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(TEST_BASE_PATH, ignore_errors=True)

    def test_audio_without_user_dir_mode(self):
        client = TEST_IO_CLIENT.audio

        # Expect an error to be raised for passing a user in userless mode
        self.assertRaises(
            ValueError,
            lambda: client.file_exists(video_id=TEST_ID, user=TEST_USER),
        )

        # Test accessing nonexistent file
        self.assertFalse(client.file_exists(video_id=TEST_ID))
        self.assertRaises(IoClientException, lambda: client.matching_paths(video_id=TEST_ID))
        self.assertRaises(IoClientException, lambda: client.read_from_disk(video_id=TEST_ID))
        self.assertRaises(IoClientException, lambda: client.delete_file(video_id=TEST_ID))

        # Test save and access file successfully
        with open(TEST_FILE_PATH, "rb") as test_file:
            audio_file = UploadFile(filename=TEST_FILE_PATH, file=test_file)
            client.save_to_disk(audio=audio_file, video_id=TEST_ID)

        self.assertTrue(client.file_exists(video_id=TEST_ID))
        self.assertEqual(1, len(client.matching_paths(video_id=TEST_ID)))

        with client.read_from_disk(video_id=TEST_ID) as read_file:
            self.assertGreater(len(read_file.read()), 0)

        client.delete_file(video_id=TEST_ID)
        self.assertFalse(client.file_exists(video_id=TEST_ID))

    def test_audio_with_user_dir_mode(self):
        client = TEST_IO_CLIENT_WITH_USER_MODE.audio

        # Expect an error to be raised for not passing a user in user mode
        self.assertRaises(
            ValueError,
            lambda: client.file_exists(video_id=TEST_ID, user=None),
        )

        # Test accessing nonexistent file
        self.assertFalse(client.file_exists(video_id=TEST_ID, user=TEST_USER))
        self.assertFalse(client.file_exists(video_id=TEST_ID, user=TEST_USER_2))
        self.assertRaises(IoClientException, lambda: client.matching_paths(video_id=TEST_ID, user=TEST_USER))
        self.assertRaises(IoClientException, lambda: client.read_from_disk(video_id=TEST_ID, user=TEST_USER))
        self.assertRaises(IoClientException, lambda: client.delete_file(video_id=TEST_ID, user=TEST_USER))

        # Test save and access file successfully
        with open(TEST_FILE_PATH, "rb") as test_file:
            audio_file = UploadFile(filename=TEST_FILE_PATH, file=test_file)
            client.save_to_disk(audio=audio_file, video_id=TEST_ID, user=TEST_USER)

        self.assertTrue(client.file_exists(video_id=TEST_ID, user=TEST_USER))
        self.assertFalse(client.file_exists(video_id=TEST_ID, user=TEST_USER_2))

        self.assertEqual(1, len(client.matching_paths(video_id=TEST_ID, user=TEST_USER)))

        with client.read_from_disk(video_id=TEST_ID, user=TEST_USER) as read_file:
            self.assertGreater(len(read_file.read()), 0)

        client.delete_file(video_id=TEST_ID, user=TEST_USER)
        self.assertFalse(client.file_exists(video_id=TEST_ID, user=TEST_USER))

    def test_video_without_user_dir_mode(self):
        client = TEST_IO_CLIENT.video

        # Expect an error to be raised for passing a user in userless mode
        self.assertRaises(
            ValueError,
            lambda: client.file_exists(video_id=TEST_ID, user=TEST_USER),
        )

        # Test accessing nonexistent file
        self.assertFalse(client.file_exists(video_id=TEST_ID))
        self.assertRaises(IoClientException, lambda: client.read_from_disk(video_id=TEST_ID))
        self.assertRaises(IoClientException, lambda: client.delete_file(video_id=TEST_ID))

        # Test save and access file successfully
        with open(TEST_FILE_PATH, "rb") as test_file:
            client.save_to_disk(video_bytes=test_file, video_id=TEST_ID)  # Save as BinaryIO
            client.save_to_disk(video_bytes=test_file.read(), video_id=TEST_ID)  # Save as bytes

        self.assertTrue(client.file_exists(video_id=TEST_ID))

        file_response = client.read_from_disk(video_id=TEST_ID)
        self.assertEqual(client.video_path(video_id=TEST_ID), file_response.path)
        self.assertGreater(file_response.chunk_size, 0)
        self.assertIn("video", file_response.media_type)

        client.delete_file(video_id=TEST_ID)
        self.assertFalse(client.file_exists(video_id=TEST_ID))

    def test_video_with_user_dir_mode(self):
        client = TEST_IO_CLIENT_WITH_USER_MODE.video

        # Expect an error to be raised for not passing a user in user mode
        self.assertRaises(
            ValueError,
            lambda: client.file_exists(video_id=TEST_ID, user=None),
        )

        # Test accessing nonexistent file
        self.assertFalse(client.file_exists(video_id=TEST_ID, user=TEST_USER))
        self.assertRaises(IoClientException, lambda: client.read_from_disk(video_id=TEST_ID, user=TEST_USER))
        self.assertRaises(IoClientException, lambda: client.delete_file(video_id=TEST_ID, user=TEST_USER))

        # Test save and access file successfully
        with open(TEST_FILE_PATH, "rb") as test_file:
            client.save_to_disk(video_bytes=test_file, video_id=TEST_ID, user=TEST_USER)  # Save as BinaryIO
            client.save_to_disk(video_bytes=test_file.read(), video_id=TEST_ID, user=TEST_USER)  # Save as bytes

        self.assertTrue(client.file_exists(video_id=TEST_ID, user=TEST_USER))
        self.assertFalse(client.file_exists(video_id=TEST_ID, user=TEST_USER_2))

        file_response = client.read_from_disk(video_id=TEST_ID, user=TEST_USER)
        self.assertEqual(client.video_path(video_id=TEST_ID, user=TEST_USER), file_response.path)
        self.assertGreater(file_response.chunk_size, 0)
        self.assertIn("video", file_response.media_type)

        client.delete_file(video_id=TEST_ID, user=TEST_USER)
        self.assertFalse(client.file_exists(video_id=TEST_ID, user=TEST_USER))
