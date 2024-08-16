import os
import pytest
from minio_client import MinioClient


@pytest.fixture
def minio_client():
    return MinioClient("localhost:9000", "minioadmin", "minioadmin")

class TestBucket:
    def test_make_bucket(self, minio_client):
        minio_client.make_bucket("test")
        assert minio_client.bucket_exists("test", verbose=False)
        minio_client.remove_bucket("test")

    def test_remove_bucket(self, minio_client):
        minio_client.make_bucket("test")
        minio_client.remove_bucket("test")
        assert not minio_client.bucket_exists("test", verbose=False)

    def test_list_buckets(self, minio_client):
        minio_client.make_bucket("test")
        buckets = minio_client.list_buckets()
        assert len(buckets) == 1
        assert buckets[0].name == "test"
        minio_client.remove_bucket("test")


class TestObject:
    def test_put_object(self, minio_client):
        minio_client.make_bucket("test")
        minio_client.put_object("test", "test_minio_client.py")
        objects = list(minio_client.list_objects("test"))
        assert len(objects) == 1
        assert objects[0].object_name == "test_minio_client.py"
        minio_client.remove_object("test", "test_minio_client.py")
        minio_client.remove_bucket("test")

    def test_put_object_diff_name(self, minio_client):
        minio_client.make_bucket("test")
        minio_client.put_object("test", "test_minio_client.py", "uploaded_test_minio_client.py")
        objects = list(minio_client.list_objects("test"))
        assert len(objects) == 1
        assert objects[0].object_name == "uploaded_test_minio_client.py"
        minio_client.remove_object("test", "uploaded_test_minio_client.py")
        minio_client.remove_bucket("test")

    def test_get_object(self, minio_client):
        minio_client.make_bucket("test")
        minio_client.put_object("test", "test_minio_client.py")
        minio_client.get_object("test", "test_minio_client.py", "downloaded_test.py")
        assert os.path.isfile("downloaded_test.py")
        os.remove("downloaded_test.py")
        minio_client.remove_object("test", "test_minio_client.py")
        minio_client.remove_bucket("test")

    def test_list_objects(self, minio_client):
        minio_client.make_bucket("test")
        minio_client.put_object("test", "test_minio_client.py")
        objects = list(minio_client.list_objects("test"))
        assert len(objects) == 1
        assert objects[0].object_name == "test_minio_client.py"
        minio_client.remove_object("test", "test_minio_client.py")
        minio_client.remove_bucket("test")

    def test_remove_object(self, minio_client):
        minio_client.make_bucket("test")
        minio_client.put_object("test", "test_minio_client.py")
        minio_client.remove_object("test", "test_minio_client.py")
        objects = list(minio_client.list_objects("test"))
        assert len(objects) == 0
        minio_client.remove_bucket("test")
