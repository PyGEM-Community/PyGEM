FROM ubuntu:latest

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo curl vim git tree python3-pip python3-venv python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

# Add non-root user 'ubuntu' to sudo group
RUN usermod -aG sudo ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu

# Switch to non-root user
USER ubuntu
WORKDIR /home/ubuntu

# Add .local/bin to PATH
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# What PyGEM branch to clone (either master or dev; see docker_pygem.yml)
ARG PYGEM_BRANCH=master

RUN git clone --branch ${PYGEM_BRANCH} https://github.com/PyGEM-Community/PyGEM.git && \
    pip install --break-system-packages -e PyGEM

# Clone the PyGEM notebooks repository, which are used for testing
RUN git clone https://github.com/PyGEM-Community/PyGEM-notebooks.git