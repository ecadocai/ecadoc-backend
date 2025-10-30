import io
import os
from types import SimpleNamespace

import pytest
from fastapi import UploadFile

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from modules.profile.service import ProfileService


class ProfileDBStub:
    def __init__(self):
        self.updated = None
        self.user = SimpleNamespace(
            id=1,
            firstname="Test",
            lastname="User",
            email="test@example.com",
            is_verified=True,
            profile_image="profile_images/existing.png",
            created_at="2025-01-01T00:00:00",
        )

    def update_user_profile(self, user_id: int, **kwargs):
        self.updated = kwargs
        return True

    def get_user_storage(self, user_id: int):
        return None

    def get_user_subscription(self, user_id: int):
        return None

    def get_user_by_id(self, user_id: int):
        return self.user if user_id == 1 else None


class AWSStub:
    def __init__(self):
        self.s3_client = object()
        self.uploaded = []

    def upload_file(self, file_obj, s3_key, content_type=None):
        self.uploaded.append((s3_key, content_type))
        return True

    def get_file_url(self, s3_key, expires_in=3600):
        return f"https://cdn.example.com/{s3_key}"

    def delete_file(self, s3_key):
        self.uploaded.append(("deleted", s3_key))
        return True


def test_upload_profile_image_stores_s3_key():
    service = ProfileService()
    service.aws_client = AWSStub()
    db_stub = ProfileDBStub()
    service.db = db_stub

    upload = UploadFile(filename="avatar.png", file=io.BytesIO(b"fake-data"), headers={"content-type": "image/png"})
    response = service.upload_profile_image(1, upload)

    stored_value = db_stub.updated["profile_image"]
    assert stored_value.startswith("profile_images/profile_1_")
    assert response["image_url"].startswith("https://cdn.example.com/profile_images/")


def test_get_user_profile_returns_resolved_image_url():
    service = ProfileService()
    aws_stub = AWSStub()
    service.aws_client = aws_stub
    db_stub = ProfileDBStub()
    service.db = db_stub

    profile = service.get_user_profile(1)

    assert profile["user"]["profile_image"] == "https://cdn.example.com/profile_images/existing.png"
