# syntax=docker/dockerfile:1@sha256:4edf897a3ffa55b89f906fc8cc78afdb3f1834cc9c7083565e611a8a7d5fe99e
# hadolint global ignore=DL3008,DL3042

FROM mitodl/ol-python-base:3.14 AS deps
LABEL maintainer="ODL DevOps <mitx-devops@mit.edu>"

# learn-ai has no app-specific apt extras; all required packages are in
# mitodl/ol-python-base:3.14.

# Install Python dependencies before copying source so this layer is only
# invalidated when lock files change.
COPY --chown=mitodl:mitodl pyproject.toml uv.lock /src/

USER mitodl
WORKDIR /src
# BuildKit cache mount keeps the uv download cache across builds.
# --no-default-groups, not --no-dev: pyproject's `default-groups = "all"` also
# selects the prek group, which --no-dev leaves in.
RUN --mount=type=cache,target=/opt/uv-cache,uid=1000,gid=1000 \
    uv sync --frozen --no-install-project --no-default-groups

FROM deps AS runtime

# Add project source.
USER root
COPY . /src
WORKDIR /src

# Write commit hash for runtime introspection.
ARG GIT_REF
RUN mkdir -p /src/static && echo "$GIT_REF" >> /src/static/hash.txt

# Run collectstatic. Fake env vars satisfy Django settings validation at
# build time; none of these values reach the runtime container. --no-sync:
# `uv run` otherwise syncs every default group (all of them, per pyproject)
# into /opt/venv first, putting the dev group into this stage.
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
    uv run --no-sync python3 manage.py collectstatic --noinput --clear

USER mitodl

EXPOSE 8888
EXPOSE 8001
ENV PORT=8001
CMD ["sh", "-c", "exec granian --interface asgi --host 0.0.0.0 --port ${PORT:-8001} main.asgi:application"]

# ─── Development target ───────────────────────────────────────────────────────
FROM runtime AS development

RUN --mount=type=cache,target=/opt/uv-cache,uid=1000,gid=1000 \
    uv sync --frozen --no-install-project

# ─── Local-dev target (ol-infrastructure local-dev k8s/Tilt stack) ────────────
FROM development AS local-dev

# runtime copies the source in as root, and Tilt's live update writes synced
# files as mitodl.
USER root
RUN chown -R mitodl:mitodl /src
USER mitodl

# Reload is set here rather than in the k8s manifest, which also runs the
# published image when there is no checkout to build. A short kill timeout stops
# a worker that won't exit from stalling a reload. Without
# RELOAD_IGNORE_WORKER_FAILURE, a save that breaks the import exits granian, and
# the restarted container comes back without the synced edits.
ENV GRANIAN_RELOAD=true \
    GRANIAN_RELOAD_IGNORE_WORKER_FAILURE=true \
    GRANIAN_WORKERS_KILL_TIMEOUT=1

# ─── Production target ───────────────────────────────────────────────────────
# MUST BE THE LAST STAGE. A build without --target produces the last stage, and
# that has to be the production image, not the development target above. Add
# new stages above this one.
FROM runtime AS production
