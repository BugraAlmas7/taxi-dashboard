#!/bin/sh
# Container entrypoint: prepare the DB, then start the server.
# `docker compose up --build` runs everything in one shot — no manual migrate/bootstrap.
set -e

echo "[entrypoint] bootstrapping database (idempotent) ..."
python manage.py bootstrap_db

# Optional one-time data load. Enable by setting RUN_SETUP_DATA=1 in .env.
# SETUP_DATA_ARGS lets you limit the months, e.g.
#   SETUP_DATA_ARGS=--months-train 2016-06:2016-06 --months-test 2017-01:2017-01
if [ "${RUN_SETUP_DATA:-0}" = "1" ]; then
  echo "[entrypoint] RUN_SETUP_DATA=1 → loading data (skips if already present) ..."
  python setup_data.py --if-empty ${SETUP_DATA_ARGS:-}
fi

echo "[entrypoint] starting Django dev server ..."
exec python manage.py runserver 0.0.0.0:8000
