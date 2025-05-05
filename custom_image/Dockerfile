FROM mambaorg/micromamba:ubuntu24.10

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

ENV PYTHONWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING='utf-8'

USER root

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    jq \
    libatlas-base-dev \
    nginx \
    openjdk-8-jdk-headless \
    unzip \ 
    wget \
    expat \
    tzdata \
    apparmor

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip

COPY environment.yml /tmp/environment.yml
COPY .condarc /opt/conda

ENV PATH=/opt/conda/bin:$PATH

RUN micromamba install -n base -f /tmp/environment.yml && \
    micromamba clean -a -y --force-pkgs-dirs && \
    rm -rf /tmp/environment.yml && \
    ln -s $(which python) /usr/bin/python

# Create MMS User Directory
COPY resources/mms/config.properties.tmp /home/model-server
ENV SKLEARN_MMS_CONFIG=/home/model-server/config.properties

# Copy execution parameters endpoint plugin for MMS
RUN mkdir -p /tmp/plugins
COPY resources/mms/endpoints-1.0.jar /tmp/plugins
RUN chmod +x /tmp/plugins/endpoints-1.0.jar

# Create a directory for the models
RUN mkdir -p /opt/ml/models && chmod +rwx /opt/ml/models && mkdir -p /opt/ml/input && chmod +rwx /opt/ml/input

# Create the serve command
COPY serve /usr/local/bin/serve
RUN chmod +x /usr/local/bin/serve

# Copy taining and serving code
RUN mkdir /opt/custom
COPY my_container /opt/custom/my_container
ENV PYTHONPATH="${PYTHONPATH}:/opt/custom"

# Setting some env vars

# Set SageMaker training environment variables
ENV SM_INPUT=/opt/ml/input
ENV SM_INPUT_TRAINING_CONFIG_FILE=/opt/ml/input/config/hyperparameters.json
ENV SM_INPUT_DATA_CONFIG_FILE=/opt/ml/input/config/inputdataconfig.json
ENV SM_CHECKPOINT_CONFIG_FILE=/opt/ml/input/config/checkpointconfig.json

LABEL com.amazonaws.sagemaker.capabilities.accept-bind-to-port=true
ENV SAGEMAKER_TRAINING_MODULE=my_container.training:main
ENV SAGEMAKER_SERVING_MODULE=my_container.serve:main

# Set SageMaker serving environment variable
ENV SM_MODEL_DIR=/opt/ml/model

EXPOSE 8080
ENV TEMP=/home/model-server/tmp

# Required label for multi-model loading
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true
