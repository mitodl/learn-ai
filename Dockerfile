# syntax=docker/dockerfile:1
# hadolint global ignore=DL3008,DL3042

FROM mitodl/ol-python-base:3.14 AS deps
LABEL maintainer="ODL DevOps <mitx-devops@mit.edu>"

# learn-ai has no app-specific apt extras; all required packages are in
# mitodl/ol-python-base:3.14.

# Install Python dependencies before copying source so this layer is only
# invalidated when lock files change.
COPY pyproject.toml uv.lock /src/
RUN chown mitodl:mitodl /src/pyproject.toml /src/uv.lock

USER mitodl
WORKDIR /src
# BuildKit cache mount keeps the uv download cache across builds.
RUN --mount=type=cache,target=/opt/uv-cache,uid=1000,gid=1000 \
    uv sync --frozen --no-install-project

# deepeval is installed without its deps to avoid heavy transitive requirements.
RUN --mount=type=cache,target=/opt/uv-cache,uid=1000,gid=1000 \
    uv pip install deepeval --no-deps

FROM deps AS final

# Add project source.
USER root
COPY . /src
WORKDIR /src

# Write commit hash for runtime introspection.
ARG GIT_REF
RUN mkdir -p /src/static && echo "$GIT_REF" >> /src/static/hash.txt

# Run collectstatic. Fake env vars satisfy Django settings validation at
# build time; none of these values reach the runtime container.
RUN UV_PROJECT_ENVIRONMENT=/opt/venv \
    DATABASE_URL="postgres://postgres:postgres@localhost:5433/postgres" \
    MITOL_SECURE_SSL_REDIRECT="False" \
    MITOL_DB_DISABLE_SSL="True" \
    MITOL_FEATURES_DEFAULT="True" \
    CELERY_TASK_ALWAYS_EAGER="True" \
    CELERY_BROKER_URL="redis://localhost:6379/4" \
    CELERY_RESULT_BACKEND="redis://localhost:6379/4" \
    MITOL_APP_BASE_URL="http://localhost:8002/" \
    MAILGUN_KEY="fake_mailgun_key" \
    MAILGUN_SENDER_DOMAIN="other.fake.site" \
    MITOL_COOKIE_DOMAIN="localhost" \
    MITOL_COOKIE_NAME="cookie_monster" \
    uv run python3 manage.py collectstatic --noinput --clear

USER mitodl

EXPOSE 8888
EXPOSE 8001
ENV PORT=8001
CMD ["granian", "--interface", "asgi", "--host", "0.0.0.0", "--port", "8001", "main.asgi:application"]
