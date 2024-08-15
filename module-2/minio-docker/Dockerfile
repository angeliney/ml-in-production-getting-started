FROM minio/minio

# Set a directory for the app
WORKDIR /usr/src/app

EXPOSE 9000
EXPOSE 9001

ENTRYPOINT ["minio"]
CMD ["server","/data","--console-address",":9001"]
