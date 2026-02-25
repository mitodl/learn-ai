FROM python:3.13.7
LABEL maintainer "ODL DevOps <mitx-devops@mit.edu>"

# Add package files, install updated node and pip
WORKDIR /tmp

# Install packages
COPY apt.txt /tmp/apt.txt
RUN apt-get update
RUN apt-get install -y $(grep -vE "^\s*#" apt.txt  | tr "\n" " ")
RUN apt-get update && apt-get install libpq-dev postgresql-client -y

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Add, and run as, non-root user.
RUN mkdir /src
RUN adduser --disabled-password --gecos "" mitodl
RUN mkdir /var/media && chown -R mitodl:mitodl /var/media

ENV UV_PROJECT_ENVIRONMENT="/opt/venv"
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml /src
COPY uv.lock /src
RUN mkdir -p /opt/venv && chown -R mitodl:mitodl /src /opt/venv

USER mitodl
WORKDIR /src
RUN uv sync --frozen --no-install-project
RUN uv pip install deepeval --no-deps

# Add project
USER root
COPY . /src
WORKDIR /src

# Generate commit hash file
ARG GIT_REF
RUN mkdir -p /src/static
RUN echo $GIT_REF >> /src/static/hash.txt

# Run collectstatic
ENV DATABASE_URL="postgres://postgres:postgres@localhost:5433/postgres"
ENV MITOL_SECURE_SSL_REDIRECT="False"
ENV MITOL_DB_DISABLE_SSL="True"
ENV MITOL_FEATURES_DEFAULT="True"
ENV CELERY_TASK_ALWAYS_EAGER="True"
ENV CELERY_BROKER_URL="redis://localhost:6379/4"
ENV CELERY_RESULT_BACKEND="redis://localhost:6379/4"
ENV MITOL_APP_BASE_URL="http://localhost:8002/"
ENV MAILGUN_KEY="fake_mailgun_key"
ENV MAILGUN_SENDER_DOMAIN="other.fake.site"
ENV MITOL_COOKIE_DOMAIN="localhost"
ENV MITOL_COOKIE_NAME="cookie_monster"
RUN uv run python3 manage.py collectstatic --noinput --clear

RUN apt-get clean && apt-get purge

USER mitodl

EXPOSE 8888
EXPOSE 8001
ENV PORT 8001
