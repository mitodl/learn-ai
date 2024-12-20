#!/usr/bin/env bash
#
# This script runs the django app

python3 manage.py collectstatic --noinput --clear

# run initial django migrations
python3 manage.py migrate --noinput

# populate cache table
python3 manage.py createcachetable

# run ONLY data migrations
RUN_DATA_MIGRATIONS=true python3 manage.py migrate --noinput

uvicorn main.asgi:application --reload --host 0.0.0.0 --port 8001

# production might best be run via a combo of gunicorn and uvicorn?
#gunicorn main.asgi:application -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001 --workers 4 --threads 2
