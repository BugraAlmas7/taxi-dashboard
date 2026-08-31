# Make the Celery app available when Django starts, so shared_task / .delay()
# use the configured broker. (Django loads this package at startup.)
from .celery import app as celery_app

__all__ = ("celery_app",)
