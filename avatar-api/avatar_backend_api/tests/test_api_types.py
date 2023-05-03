from unittest import TestCase

from avatar_backend_api.api_types import AvatarModelRequest, BaseAvatar


class TestApiTypes(TestCase):
    def test_model_request_uuid(self):
        request1 = AvatarModelRequest(avatar=BaseAvatar.Jennifer_355_9415)
        request2 = AvatarModelRequest(avatar=BaseAvatar.Jennifer_355_9415)

        self.assertNotEqual(request1.video_id, request2.video_id)
