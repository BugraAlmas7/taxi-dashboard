"""
anomaly/tasks.py
Celery tasks — the HEAVY streaming/ML work that runs on the separate `worker`
container instead of blocking the `web` UI process.

Flow:
  web  ── enqueue run_streaming_pipeline.delay(overrides) ──▶  Redis  ──▶  worker
  worker runs the Orchestrator loop (IF clean → XGB/Chronos update), writing:
     • model files to models_update/ + finetune/ (shared volume, web reads them)
     • one PipelineWindowResult row per window to Postgres (web polls them)
  web  ── set the Redis "stop" flag ──▶  worker's loop stops cooperatively.

Nothing here talks to the web process in memory — all hand-off is via the shared
DB + the shared code/model volume, which is exactly what makes the split clean.
"""
import os

import redis
from celery import shared_task

from .streaming.config import PipelineConfig
from .streaming.orchestrator import Orchestrator

_REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://redis:6379/0")


def _redis():
    return redis.Redis.from_url(_REDIS_URL)


def _stop_key(run_name):
    return f"stream:stop:{run_name}"


def _running_key(run_name):
    return f"stream:running:{run_name}"


@shared_task(name="anomaly.run_streaming_pipeline")
def run_streaming_pipeline(overrides=None):
    """Run the live streaming pipeline to exhaustion (or cfg.max_windows) on the
    worker. `overrides` is a plain dict of PipelineConfig fields (JSON-safe), so
    it can travel through the broker."""
    overrides = overrides or {}
    cfg = PipelineConfig(**overrides)

    r = _redis()
    run = cfg.run_name
    r.delete(_stop_key(run))                 # clear any stale stop flag
    r.set(_running_key(run), "1")

    def _should_stop():
        return r.exists(_stop_key(run)) == 1

    try:
        Orchestrator(cfg).run(should_stop=_should_stop)
    finally:
        r.delete(_running_key(run))
        r.delete(_stop_key(run))
    return {"run_name": run, "status": "finished"}


def request_stop(run_name="stream"):
    """Set the cooperative stop flag; the worker loop ends between windows."""
    _redis().set(_stop_key(run_name), "1")


def is_running(run_name="stream"):
    return _redis().exists(_running_key(run_name)) == 1
