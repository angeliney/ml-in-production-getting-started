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