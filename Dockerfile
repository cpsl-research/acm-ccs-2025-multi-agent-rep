FROM nvidia/cuda:12.0.1-devel-ubuntu22.04

# Arguments
ARG UV_CACHE_DIR=/tmp/uv-cache
ARG DEBIAN_FRONTEND=noninteractive

# Environment variables
ENV PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

# Install python and pip
RUN set -xe \
    && apt-get update \
    && apt-get -y install curl ca-certificates python3-pip python3.10-venv ffmpeg libsm6 libxext6 wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"
COPY --from=ghcr.io/astral-sh/uv:0.8.2 /uv /uvx /bin/

# Create a non-root user and set up cache directory (only once)
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /home/appuser/.cache/uv \
    && chown -R appuser:appuser /home/appuser/.cache

# Copy only dependency files first for better caching
COPY uv.lock pyproject.toml /tmp/code/

# Copy the submodules over
COPY ./submodules /tmp/code/submodules
WORKDIR /tmp/code

# Set up UV caching and install as root
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
    uv sync --cache-dir=${UV_CACHE_DIR}

USER appuser
WORKDIR /home/appuser/code

# Copy the rest of the project
COPY --chown=appuser:appuser . /home/appuser/code

# Healthcheck: verify Python is available
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python3 -c "import sys; sys.exit(0)"