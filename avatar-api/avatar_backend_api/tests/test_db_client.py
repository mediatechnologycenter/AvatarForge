from unittest import IsolatedAsyncioTestCase

from mtc_api_utils.api_types import FirebaseUser

from avatar_backend_api.api_types import VideoMetadata, AvatarModel, BaseAvatar
from avatar_backend_api.clients.db_client import AvatarDbClient, AvatarDbException

TEST_DB_DIR = "/tmp/test-dbs/avatar-db.json"

TEST_USER = FirebaseUser.example()

TEST_METADATA = VideoMetadata(
    video_id="test-id",
    audio_name="test-video-name",
    avatar=BaseAvatar.Jennifer_355_9415,
    avatar_model=AvatarModel.motion_gan,
)

TEST_METADATA_2 = VideoMetadata(
    video_id=TEST_METADATA.video_id + "-2",
    audio_name=TEST_METADATA.audio_name + "-2",
    avatar=TEST_METADATA.avatar,
    avatar_model=TEST_METADATA.avatar_model,
)

TEST_DB_CLIENT = AvatarDbClient(db_filepath=TEST_DB_DIR)


class TestDbClient(IsolatedAsyncioTestCase):

    def setUp(self) -> None:
        TEST_DB_CLIENT.db.drop_tables()

    def tearDown(self) -> None:
        self.setUp()

    def test_db_client(self):
        # Test access nonexistent entries & error handling
        self.assertFalse(TEST_DB_CLIENT.video_exists(video_id=TEST_METADATA.video_id, user=TEST_USER))
        self.assertEqual(0, len(TEST_DB_CLIENT.list_videos(user=TEST_USER)))
        self.assertRaises(AvatarDbException, lambda: TEST_DB_CLIENT.get_video(video_id=TEST_METADATA.video_id, user=TEST_USER))
        self.assertRaises(AvatarDbException, lambda: TEST_DB_CLIENT.delete_video(TEST_METADATA_2.video_id, user=TEST_USER))
        self.assertRaises(AvatarDbException, lambda: TEST_DB_CLIENT.get_user_from_id(video_id=TEST_METADATA.video_id))

        # Test successful manipulation of db
        TEST_DB_CLIENT.insert_video(video_metadata=TEST_METADATA, user=TEST_USER)
        self.assertEqual(1, len(TEST_DB_CLIENT.list_videos(user=TEST_USER)))
        self.assertEqual(TEST_METADATA, TEST_DB_CLIENT.get_video(TEST_METADATA.video_id, user=TEST_USER))
        self.assertEqual(TEST_METADATA, TEST_DB_CLIENT.get_video(video_id=TEST_METADATA.video_id, user=TEST_USER))

        TEST_METADATA.inference_completed = True
        TEST_DB_CLIENT.upsert_video(video_metadata=TEST_METADATA, user=TEST_USER)
        self.assertEqual(TEST_METADATA, TEST_DB_CLIENT.get_video(video_id=TEST_METADATA.video_id, user=TEST_USER))

        TEST_DB_CLIENT.insert_video(video_metadata=TEST_METADATA_2, user=TEST_USER)
        self.assertEqual(2, len(TEST_DB_CLIENT.list_videos(user=TEST_USER)))
        self.assertEqual(TEST_METADATA_2, TEST_DB_CLIENT.get_video(video_id=TEST_METADATA_2.video_id, user=TEST_USER))

        self.assertEqual(TEST_METADATA_2, TEST_DB_CLIENT.delete_video(TEST_METADATA_2.video_id, user=TEST_USER))
        self.assertEqual(1, len(TEST_DB_CLIENT.list_videos(user=TEST_USER)))
        self.assertEqual(TEST_METADATA, TEST_DB_CLIENT.get_video(TEST_METADATA.video_id, user=TEST_USER))
        self.assertEqual(TEST_METADATA, TEST_DB_CLIENT.delete_video(TEST_METADATA.video_id, user=TEST_USER))
