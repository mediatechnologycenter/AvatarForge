# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import unittest
from enum import Enum
from http import HTTPStatus
from time import sleep

import httpx

from avatar_backend_api.api_types import AvatarModelRequest, BaseAvatar
from neural_voice_backend_api.app import app
from neural_voice_backend_api.tests.integration_test_client import AvatarIntegrationTestClient


class TestFile(Enum):
    value: str
    test_mp3 = "test.mp3"
    ping_wav = "ping.wav"
    voice_wav = "voice_test.wav"
    frontend_wav = "Test-2.wav"


test_dir = os.path.dirname(os.path.realpath(__file__))
test_file_name = TestFile.frontend_wav.value
test_file_path = os.path.join(test_dir, test_file_name)

TEST_REQUEST = AvatarModelRequest(
    video_id="id",
    avatar=BaseAvatar.Jennifer_355_9415,
)


class TestIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.client = AvatarIntegrationTestClient(test_app=app)

        self.client.wait_for_service_readiness()
        try:
            self.client.delete_video(TEST_REQUEST.video_id)
        except httpx.HTTPStatusError as e:
            self.assertEqual(e.response.status_code, HTTPStatus.NOT_FOUND)

        print("Integration Test Setup Completed")

    def tearDown(self) -> None:
        self.setUp()

    def test_inference(self):
        with open(test_file_path, "rb") as test_file:
            initial_video_ids = self.client.list_video_ids()
            self.assertNotIn(TEST_REQUEST.video_id, initial_video_ids)

            try:
                self.client.delete_video(TEST_REQUEST.video_id)
                self.fail("Expected delete to fail because video is not present")
            except httpx.HTTPStatusError as e:
                self.assertEqual(e.response.status_code, HTTPStatus.NOT_FOUND)

            self.assertEqual(TEST_REQUEST, self.client.post_audio(audio=test_file, metadata=TEST_REQUEST))

            while True:
                status_code, file_bytes = self.client.get_video(video_id=TEST_REQUEST.video_id)
                self.assertIn(status_code, [HTTPStatus.OK, HTTPStatus.PROCESSING])

                if status_code == HTTPStatus.OK:
                    break
                else:
                    sleep(1)

            print("Test Delete Video")
            self.client.delete_video(video_id=TEST_REQUEST.video_id)
