from minio import Minio


class MinioClient:
    def __init__(self, host, access_key, secret_key):
        self.client = Minio(host, access_key=access_key, secret_key=secret_key, secure=False)

    def bucket_exists(self, bucket_name, verbose=True):
        found = self.client.bucket_exists(bucket_name)
        if not found and verbose:
            print("Bucket", bucket_name, "doesn't exist")
        return found
    
    def make_bucket(self, bucket_name):
        found = self.bucket_exists(bucket_name, verbose=False)
        if not found:
            self.client.make_bucket(bucket_name)
            print("Created bucket", bucket_name)
        else:
            print("Bucket", bucket_name, "already exists")

    def list_buckets(self):
        buckets = self.client.list_buckets()
        for bucket in buckets:
            print(bucket.name, bucket.creation_date)
        return buckets

    def remove_bucket(self, bucket_name):
        found = self.bucket_exists(bucket_name)
        if found:
            self.client.remove_bucket(bucket_name)
            print("Removed bucket", bucket_name)

    def put_object(self, bucket_name, source_file, destination_file=None, **kwargs):
        if destination_file is None:
            destination_file = source_file
        found = self.bucket_exists(bucket_name)
        if found:
            self.client.fput_object(bucket_name, destination_file, source_file, **kwargs)
            print("Uploaded", source_file, "to", bucket_name, "as", destination_file)
    
    def get_object(self, bucket_name, filename, destination_file=None, **kwargs):
        if destination_file is None:
            destination_file = filename
        found = self.bucket_exists(bucket_name)
        if found:
            self.client.fget_object(bucket_name, filename, destination_file, **kwargs)
            print("Downloaded", filename, "from", bucket_name, "as", destination_file)

    def list_objects(self, bucket_name, **kwargs):
        found = self.bucket_exists(bucket_name)
        if found:
            objects = self.client.list_objects(bucket_name, **kwargs)
            return objects
    
    def remove_object(self, bucket_name, filename, **kwargs):
        found = self.client.bucket_exists(bucket_name)
        if found:
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
    objects = minio_client.list_objects("test")
    for obj in objects:
        print(obj)
    minio_client.get_object("test", "requirements.txt", "downloaded_requirements.txt")
    minio_client.remove_object("test", "requirements.txt")
    minio_client.remove_bucket("test")
    

if __name__ == "__main__":
    main()