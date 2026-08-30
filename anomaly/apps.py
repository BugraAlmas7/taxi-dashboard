"""
apps.py — Django AppConfig with model warm-up at server startup.

Preloads the heavy forecast models (Chronos / TimesFM + LoRA adapters) into the
in-process cache when the server boots, so the FIRST forecast in a live demo
doesn't hang on a multi-second cold-start.

Design notes:
  • Runs in a BACKGROUND daemon thread → server boot is not blocked; the models
    finish loading a few seconds later while the page is already up.
  • Only warms up when actually SERVING (runserver / gunicorn), NOT during
    migrate / makemigrations / shell / collectstatic — those would pay the load
    cost for nothing (and can fail without a GPU).
  • Guards runserver's autoreloader (RUN_MAIN) so it doesn't load twice.
  • Fully guarded: any warm-up failure is swallowed so it can never crash boot.

⚠️ Set `name` below to YOUR app's import path (the folder that holds
   forecast_calculation.py — same package as views.py). It looks like "anomaly"
   from your templates; change if different. Then register this AppConfig:
   either Django picks it up automatically if this file is <app>/apps.py, or set
   default_app_config / the INSTALLED_APPS entry to
   "<app>.apps.ForecastAppConfig".
"""
import os
import sys
import threading

from django.apps import AppConfig


class ForecastAppConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "anomaly"          # ← CHANGE to your app package name if different

    def ready(self):
        # 1) ONLY warm up / stream when actually SERVING (runserver / gunicorn /
        #    uvicorn / daphne). Any other django.setup() — a management command,
        #    a standalone script like training/train_if.py, migrate, shell — must
        #    NOT trigger the heavy model warm-up.
        argv0 = os.path.basename(sys.argv[0] or "")
        serving = ("runserver" in sys.argv) or \
                  any(argv0.startswith(s) for s in ("gunicorn", "uvicorn", "daphne"))
        if not serving:
            return

        # 2) runserver spawns two processes (reloader + worker); only run in the worker
        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") != "true":
            return

        # ── (a) warm up the heavy models ─────────────────────────────────────
        if os.environ.get("WARMUP_MODELS", "1") not in ("0", "false", "False"):
            def _warm():
                try:
                    from .forecast_calculation import warmup_models
                    warmup_models()
                except Exception as e:
                    print(f"[warmup] disabled ({type(e).__name__}: {e})")
            threading.Thread(target=_warm, name="forecast-warmup", daemon=True).start()
            print("[warmup] background model preload started")

        # ── (b) OPTIONAL auto-run the streaming pipeline on boot ─────────────
        # OFF by default so `runserver` starts clean/quiet — start the pipeline
        # from the Live page's "Start Pipeline" button instead. To auto-run on
        # boot (headless/demo), set STREAM=1.
        if os.environ.get("STREAM", "0") not in ("0", "false", "False"):
            def _stream():
                try:
                    from .streaming import PipelineConfig, run_pipeline
                    cfg = PipelineConfig(
                        models=[m for m in os.environ.get("STREAM_MODELS", "xgboost").split(",") if m],
                        mode=os.environ.get("STREAM_MODE", "finetune"),   # first window: train → then finetune
                        source=os.environ.get("STREAM_SOURCE", "trip2017"),
                        start=os.environ.get("STREAM_START", "2017-01-02 00:00"),
                        max_windows=int(os.environ.get("STREAM_MAX_WINDOWS", "0")),
                        tick_sleep=float(os.environ.get("STREAM_SLEEP", "0.4")),
                        run_name="auto",
                    )
                    print(f"[stream] auto pipeline start · models={cfg.models} mode={cfg.mode}")
                    run_pipeline(cfg)
                    print("[stream] auto pipeline finished")
                except Exception as e:
                    print(f"[stream] disabled ({type(e).__name__}: {e})")
            threading.Thread(target=_stream, name="stream-pipeline", daemon=True).start()
