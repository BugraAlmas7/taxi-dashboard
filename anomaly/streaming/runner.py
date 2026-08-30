"""
streaming/runner.py — start/stop the streaming pipeline ON DEMAND from the UI.

The server no longer runs the pipeline on boot. The Live page's "Start Pipeline"
button hits an endpoint that calls start(); "Durdur" calls stop(). One pipeline at
a time; a cooperative stop flag ends it cleanly between windows.
"""
import threading

from .config import PipelineConfig
from .orchestrator import Orchestrator

_lock = threading.Lock()
_state = {"thread": None, "stop": None, "running": False}


def start(**overrides):
    """Spawn the pipeline in a background daemon thread. Returns True if started,
    False if one is already running."""
    with _lock:
        if _state["running"]:
            return False
        stop_evt = threading.Event()
        opts = dict(models=["xgboost"], mode="finetune", run_name="auto", tick_sleep=0.4)
        opts.update(overrides)
        cfg = PipelineConfig(**opts)

        def _run():
            try:
                Orchestrator(cfg).run(should_stop=stop_evt.is_set)
            except Exception as e:
                print(f"[stream] error: {type(e).__name__}: {e}")
            finally:
                _state["running"] = False
                print("[stream] stopped")

        t = threading.Thread(target=_run, name="stream-pipeline", daemon=True)
        _state.update(thread=t, stop=stop_evt, running=True)
        t.start()
        print(f"[stream] started · models={cfg.models} mode={cfg.mode}")
        return True


def stop():
    with _lock:
        if _state["stop"]:
            _state["stop"].set()
        _state["running"] = False


def status():
    return {"running": bool(_state["running"])}
