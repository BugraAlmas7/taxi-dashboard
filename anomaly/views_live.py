"""
views_live.py  — live flowing forecast panel
Paste these two views into your app's views.py (or keep as a module and import),
and wire the two URLs (see urls_live_snippet.py).

  • live_page  → renders the 5-panel flowing page (templates/anomaly/live.html)
  • live_tick  → JSON: trailing actual + pure H-step-ahead forecast for one panel

The forecast LEADS the actual: each tick advances a simulated "now"; the chart
draws the forecast extending past now, and as now advances the actual catches up
underneath it — the whole point of the demo.
"""
from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from .streaming.live_forecast import live_forecast

# models selectable in the live panel (live variants + a static baseline to compare)
LIVE_MODELS = [
    ("xgboost_stream", "XGBoost (Live)"),
    ("chronos_stream", "Chronos-2 (Live)"),
    ("xgboost", "XGBoost (baseline)"),
    ("chronos_ft_optuna", "Chronos-2 (Optuna)"),
]
# only demand + revenue are meaningful to forecast live
LIVE_METRICS = ["sefer", "total_amount"]


@login_required
def live_page(request):
    ctx = {
        "models": LIVE_MODELS,
        "metrics": LIVE_METRICS,
        "default_model": "xgboost_stream",
        "resolution": "1h",
        "vendor": "hepsi",
        # TEST on 2017 (the model is trained on 2015-16 only, prewarmed from the
        # 2016 tail) → nothing shown was in training. 2017 is a pure holdout.
        "start": "2017-01-01 00:00",
        "horizon": 12,          # bins ahead (1h → 12 hours lead)
    }
    return render(request, "anomaly/live.html", ctx)


@login_required
def live_retrain(request):
    """Display-driven nightly retrain: clean one day + update the live model,
    synchronously, so the page can pause and wait for it."""
    from .streaming.live_session import SESSION
    p = request.GET
    model = "chronos" if p.get("model") == "chronos_stream" else "xgboost"
    res = SESSION.retrain(
        day=p.get("day"), metrics=LIVE_METRICS, model=model,
        mode=p.get("mode", "finetune"), resolution=p.get("resolution", "1h"),
        vendor=p.get("vendor", "hepsi"), source="trip2017",
        start=p.get("start", "2017-01-01 00:00"))
    return JsonResponse(res)


@login_required
def pipeline_control(request):
    """Start/stop the streaming pipeline on demand (Live page button)."""
    from .streaming import runner
    action = request.GET.get("action", "status")
    if action == "start":
        started = runner.start()
        return JsonResponse({"running": True, "started": started})
    if action == "stop":
        runner.stop()
        return JsonResponse({"running": False})
    return JsonResponse(runner.status())


@login_required
def pipeline_status(request):
    """Latest Isolation-Forest cleaning stats from the background pipeline —
    so the Live page can show anomalies being detected/removed in real time."""
    from .models import PipelineWindowResult
    run = request.GET.get("run", "auto")
    # one row per window (cleaning stats are identical across a window's models)
    rows = (PipelineWindowResult.objects
            .filter(run_name=run).order_by("-window_index")
            .values("window_index", "ts_start", "n_raw", "n_anomaly",
                    "n_clean", "anom_rate")[:40])
    seen, out = set(), []
    for r in rows:
        if r["window_index"] in seen:
            continue
        seen.add(r["window_index"])
        out.append(r)
    out.reverse()                       # oldest → newest for the sparkline
    latest = out[-1] if out else None
    return JsonResponse({"windows": out, "latest": latest})


@login_required
def live_tick(request):
    p = request.GET
    data = live_forecast(
        model_key=p.get("model", "xgboost_stream"),
        metric=p.get("metric", "sefer"),
        resolution=p.get("resolution", "1m"),
        vendor=p.get("vendor", "hepsi"),
        now=p.get("now", "2017-01-02 00:00"),
        horizon=int(p.get("horizon", 60)),
        show_bins=int(p.get("show", 180)),
        how=p.get("how", "mean"),
    )
    return JsonResponse(data)
