FROM python:3.12.4-slim

# set a directory for the app
WORKDIR /usr/src/app

# copy requirements file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the files to the container
COPY app.py .
COPY templates templates/

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "./app.py"]
