FROM nvidia/cuda:10.2-base-ubuntu18.04

LABEL maintainer Damiano Brunori <brunori@diag.uniroma1.it>

# Install basic utilities #apt-utils
RUN apt-get update && \
    apt-get install -y wget --no-install-recommends \
    sudo build-essential \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && rm ~/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

COPY . .

RUN conda env create -f requirements.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "rl_uavs", "/bin/bash", "-c"]

# Entrypoint doesn't start a shell section. This code run when container is started 
ENTRYPOINT  ["conda", "run","--no-capture-output","-n","rl_uavs"]
