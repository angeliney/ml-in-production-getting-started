FROM ubuntu:oracular-20240617

RUN  apt-get update \
  && apt-get install -y wget

# Set a directory for the app
WORKDIR /usr/src/app

# Copy script
COPY run.sh .
COPY data data/

# tell the ports for API and WebUI
EXPOSE 9000 30000-59999

# Make a data container
RUN chmod +x ./run.sh
CMD ["bash", "./run.sh"]
