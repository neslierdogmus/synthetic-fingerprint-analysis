FROM ubuntu:22.04

LABEL maintainer="nesli <neslierdogmus@iyte.edu.tr>"

# Install system dependencies
RUN apt-get update --fix-missing && \
    apt-get install -y python3-pip libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install any python packages you need
COPY ./requirements.txt /tmp/

RUN python3 -m pip install -r /tmp/requirements.txt

# Set the project directory
ENV PROJECT_DIR=/home/finperprint-analysis
RUN mkdir $PROJECT_DIR && \
    mkdir $PROJECT_DIR/src && \
    mkdir $PROJECT_DIR/foe && \
    mkdir $PROJECT_DIR/fc && \
    mkdir $PROJECT_DIR/fmd
WORKDIR $PROJECT_DIR

# add src folder to PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:$PROJECT_DIR/foe/src:$PROJECT_DIR/fc/src:$PROJECT_DIR/fmd/src"

# Set default command
CMD [ "/bin/bash" ]
