from minio import Minio


class MinioClient:
    def __init__(self, host, access_key, secret_key):
        self.client = Minio(host, access_key=access_key, secret_key=secret_key, secure=False)

    def make_bucket(self, bucket_name):
        found = self.client.bucket_exists(bucket_name)
        if not found:
            self.client.make_bucket(bucket_name)
            print("Created bucket", bucket_name)
        else:
            print("Bucket", bucket_name, "already exists")

    def list_buckets(self):
        buckets = self.client.list_buckets()
        for bucket in buckets:
            print(bucket.name, bucket.creation_date)

    def remove_bucket(self, bucket_name):
        # found = self.client.bucket_exists(bucket_name)
        # if not found:
        #     print("Bucket", bucket_name, "doesn't exist")
        # else:
        self.client.remove_bucket(bucket_name)
        print("Removed bucket", bucket_name)

    def put_object(self, bucket_name, source_file, destination_file=None, **kwargs):
        if destination_file is None:
            destination_file = source_file
        found = self.client.bucket_exists(bucket_name)
        if not found:
            print("Bucket", bucket_name, "doesn't exist")
        else:
            self.client.fput_object(bucket_name, destination_file, source_file, **kwargs)
            print("Uploaded", source_file, "to", bucket_name, "as", destination_file)
    
    def get_object(self, bucket_name, filename, destination_file=None, **kwargs):
        if destination_file is None:
            destination_file = filename
        found = self.client.bucket_exists(bucket_name)
        if not found:
            print("Bucket", bucket_name, "doesn't exist")
        else:
            self.client.fget_object(bucket_name, filename, destination_file, **kwargs)
            print("Downloaded", filename, "from", bucket_name, "as", destination_file)

    def list_objects(self, bucket_name, **kwargs):
        objects = self.client.list_objects(bucket_name, **kwargs)
        for obj in objects:
            print(obj)
    
    def remove_object(self, bucket_name, filename, **kwargs):
        found = self.client.bucket_exists(bucket_name)
        if not found:
            print("Bucket", bucket_name, "doesn't exist")
        else:
            self.client.remove_object(bucket_name, filename, **kwargs)
            print("Removed", filename, "from", bucket_name)


def main():
    minio_client = MinioClient("localhost:9000", "minioadmin", "minioadmin")
    print("Client connected")
    minio_client.make_bucket("test")
    minio_client.list_buckets()
    minio_client.remove_bucket("test")
    minio_client.list_buckets()
    minio_client.make_bucket("test")
    minio_client.put_object("test", "requirements.txt")
    minio_client.list_objects("test")
    minio_client.get_object("test", "requirements.txt", "downloaded_requirements.txt")
    

if __name__ == "__main__":
    main()