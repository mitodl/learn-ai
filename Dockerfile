# hadolint global ignore=DL3008,SC2046
FROM python:3.13.2
LABEL org.opencontainers.image.authors="ODL DevOps <mitx-devops@mit.edu>"

# Set shell to bash with pipefail
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Add package files, install updated node and pip
WORKDIR /tmp

# Install packages
COPY apt.txt /tmp/apt.txt
RUN apt-get update \
    && apt-get install -y \
    --no-install-recommends \
    libpq-dev \
    postgresql-client \
    $(grep -vE '^\s*#' apt.txt  | tr '\n' ' ') \
    && apt-get clean \
    && apt-get purge \
    && rm -rf /var/lib/apt/lists/*

# Add, and run as, non-root user.
RUN mkdir /src \
    && adduser --disabled-password --gecos "" mitodl \
    && mkdir /var/media && chown -R mitodl:mitodl /var/media

## Set some poetry config
ENV  \
  PYTHON_UNBUFFERED=1 \
  POETRY_VERSION=1.8.5 \
  POETRY_VIRTUALENVS_CREATE=true \
  POETRY_CACHE_DIR='/tmp/cache/poetry' \
  POETRY_HOME='/home/mitodl/.local' \
  VIRTUAL_ENV="/opt/venv"
ENV PATH="$VIRTUAL_ENV/bin:$POETRY_HOME/bin:$PATH"

# Install poetry
RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml /src
COPY poetry.lock /src
RUN chown -R mitodl:mitodl /src && \
    mkdir ${VIRTUAL_ENV} && \
    chown -R mitodl:mitodl ${VIRTUAL_ENV}

## Install poetry itself, and pre-create a venv with predictable name
USER mitodl
WORKDIR /src
RUN python3 -m venv $VIRTUAL_ENV && \
    poetry install

# Add project
USER root
COPY . /src
WORKDIR /src

# Generate commit hash file
ARG GIT_REF
ARG RELEASE_VERSION
RUN mkdir -p /src/static \
    && echo "{\"version\": \"$RELEASE_VERSION\", \"hash\": \"$GIT_REF\"}" >> /src/static/hash.txt

USER mitodl

EXPOSE 8888
EXPOSE 8001
ENV PORT=8001
