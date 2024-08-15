# Module 2 practice 

## Deploy minio
### Local
```sh
wget https://dl.min.io/server/minio/release/linux-amd64/minio
chmod +x minio
./minio server /data
```
Go to `http://172.21.46.1:39539` or the printed URL to see the WebUI.

### Docker
```sh
cd minio-docker
docker build . -t angeliney/minio-docker:latest
docker run -p 9001:9001 angeliney/minio-docker:latest
```
Go to `http://127.0.0.1:9001` to access the WebUI.