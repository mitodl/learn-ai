FROM python:3.13.5
LABEL maintainer "ODL DevOps <mitx-devops@mit.edu>"

# Add package files, install updated node and pip
WORKDIR /tmp

# Install packages
COPY apt.txt /tmp/apt.txt
RUN apt-get update
RUN apt-get install -y $(grep -vE "^\s*#" apt.txt  | tr "\n" " ")
RUN apt-get update && apt-get install libpq-dev postgresql-client -y

# pip
RUN curl --silent --location https://bootstrap.pypa.io/get-pip.py | python3 -

# Add, and run as, non-root user.
RUN mkdir /src
RUN adduser --disabled-password --gecos "" mitodl
RUN mkdir /var/media && chown -R mitodl:mitodl /var/media

## Set some poetry config
ENV  \
  POETRY_VERSION=1.7.1 \
  POETRY_VIRTUALENVS_CREATE=true \
  POETRY_CACHE_DIR='/tmp/cache/poetry' \
  POETRY_HOME='/home/mitodl/.local' \
  VIRTUAL_ENV="/opt/venv"
ENV PATH="$VIRTUAL_ENV/bin:$POETRY_HOME/bin:$PATH"

# Install poetry
RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml /src
COPY poetry.lock /src
RUN chown -R mitodl:mitodl /src
RUN mkdir ${VIRTUAL_ENV} && chown -R mitodl:mitodl ${VIRTUAL_ENV}

## Install poetry itself, and pre-create a venv with predictable name
USER mitodl
RUN curl -sSL https://install.python-poetry.org \
  | \
  POETRY_VERSION=${POETRY_VERSION} \
  POETRY_HOME=${POETRY_HOME} \
  python3 -q
WORKDIR /src
RUN python3 -m venv $VIRTUAL_ENV
RUN poetry install

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
RUN python3 manage.py collectstatic --noinput --clear

RUN apt-get clean && apt-get purge

USER mitodl

EXPOSE 8888
EXPOSE 8001
ENV PORT 8001
