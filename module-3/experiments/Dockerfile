FROM python:3.12.4-slim

# set a directory for the app
WORKDIR /usr/src/app

# Install GCP
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg curl
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN apt-get update && apt-get -y install google-cloud-cli

# set a directory for the app
WORKDIR /usr/src/app

# Copy the keys to the container
COPY gcp_keys.json .

# Authenticate with the keys
RUN gcloud auth activate-service-account --key-file=gcp_keys.json

# copy requirements file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the files to the container
COPY experiments.py .
COPY .env .

# run the command
CMD ["python", "./experiments.py"]
