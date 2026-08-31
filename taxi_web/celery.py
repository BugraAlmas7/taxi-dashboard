"""
taxi_web/celery.py
Celery application for the project. The heavy streaming/ML work (the live
pipeline: Isolation-Forest cleaning + XGBoost/Chronos updates) is enqueued from
the `web` container and executed on a SEPARATE `worker` container, so the UI is
never blocked by model training.

Broker + result backend = Redis (the `redis` service in docker-compose).
"""
import os

import django
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")

# Populate the Django app registry BEFORE autodiscover imports anomaly/tasks.py
# (which imports models). Without this a standalone celery worker — not launched
# by Django — raises AppRegistryNotReady on task import.
django.setup()

# redis://redis:6379/0 by default (the compose service name is `redis`).
BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")

app = Celery("taxi_web", broker=BROKER_URL, backend=BROKER_URL)

# all celery config keys are read from Django settings with a CELERY_ prefix
app.config_from_object("django.conf:settings", namespace="CELERY")

# auto-discover tasks.py in every installed app (finds anomaly/tasks.py)
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f"[celery] request: {self.request!r}")
