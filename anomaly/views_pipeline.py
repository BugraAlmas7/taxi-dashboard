"""
anomaly/views_pipeline.py
Thin HTTP layer over the Celery streaming pipeline. The web container only
ENQUEUES / STOPS / READS STATUS — the actual work runs on the `worker`
container. This is the "web ↔ worker" seam of the multi-container split.

  POST /pipeline/start   → enqueue run_streaming_pipeline on the worker
  POST /pipeline/stop    → set the cooperative stop flag (worker ends cleanly)
  GET  /pipeline/status  → running flag + the latest per-window results (from DB)
"""
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required

from .models import PipelineWindowResult
from . import tasks


@login_required
@require_POST
def pipeline_start(request):
    p = request.POST
    run = p.get("run_name", "stream")
    if tasks.is_running(run):
        return JsonResponse({"ok": False, "error": "already running", "run_name": run})

    overrides = {"run_name": run}
    # optional knobs from the form (all have PipelineConfig defaults)
    if p.get("models"):
        overrides["models"] = [m.strip() for m in p["models"].split(",") if m.strip()]
    if p.get("mode"):
        overrides["mode"] = p["mode"]
    if p.get("source"):
        overrides["source"] = p["source"]
    if p.get("start"):
        overrides["start"] = p["start"]
    if p.get("max_windows"):
        overrides["max_windows"] = int(p["max_windows"])
    if p.get("tick_sleep"):
        overrides["tick_sleep"] = float(p["tick_sleep"])

    async_result = tasks.run_streaming_pipeline.delay(overrides)
    return JsonResponse({"ok": True, "task_id": async_result.id, "run_name": run})


@login_required
@require_POST
def pipeline_stop(request):
    run = request.POST.get("run_name", "stream")
    tasks.request_stop(run)
    return JsonResponse({"ok": True, "run_name": run, "stopping": True})


@login_required
def pipeline_status(request):
    run = request.GET.get("run_name", "stream")
    rows = list(
        PipelineWindowResult.objects
        .filter(run_name=run)
        .order_by("-window_index", "model_name")
        .values("window_index", "ts_start", "n_raw", "n_anomaly", "n_clean",
                "anom_rate", "metric", "model_name", "mode", "status", "wape", "mae")[:60]
    )
    return JsonResponse({"run_name": run, "running": tasks.is_running(run), "rows": rows})
