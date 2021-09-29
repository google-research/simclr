# or tensorflow/tensorflow:2.4.2-gpu-jupyter for GPU use cases
ARG base_image=tensorflow/tensorflow:2.6.0-jupyter
FROM $base_image

# install apt-get packages
# set apt-get to non-interactive so it doesn't prompt for configuration (will use the defaults instead)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y \
    build-essential \
    curl \
    ca-certificates \
    git \
    htop \
    moreutils \
    screen \
    software-properties-common \
    vim \
    wget \
    zip \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install the Google Cloud SDK, which gives us `gcloud`, `gsutil`, `bq`.
# See https://cloud.google.com/sdk/downloads#apt-get
RUN apt-get update && apt-get -y install apt-transport-https && \
    export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb https://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" \
        | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update && \
    apt-get install -y google-cloud-sdk \
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

COPY ./requirements.txt /requirements.txt
# jedi breaks ipython autocomplete. uninstall it (https://stackoverflow.com/a/65579764/3893740)
RUN pip install --upgrade pip && \
    pip install -r /requirements.txt && \
    pip uninstall jedi --yes && \
    rm -r ~/.cache/pip

ENV TERM=xterm

# Assumes this repo is mounted as shared volume at /simclr.
# Make this the default working directory and add to PYTHONPATH.
WORKDIR /simclr
ENV PYTHONPATH="/simclr:$PYTHONPATH"

CMD ["/bin/bash"]
