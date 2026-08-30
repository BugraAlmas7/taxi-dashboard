# Live Streaming Pipeline — Architecture & Integration

A live pipeline added to the NYC taxi dashboard: it continuously pulls data from
the test dataset, cleans raw records in sliding windows with **Isolation
Forest**, automatically updates the forecast models (**XGBoost** + **Chronos-2**)
with the cleaned data, and writes each window's result to a DB table; the
dashboard reads it and shows it live.

## Flow (at a glance)

```
StreamSimulator ──batch──▶ SlidingWindow ──window──▶ IsolationForestCleaner
   (sefer_2017 /              (W=1000,                    (raw trip level:
    sefer_egitim,              step=200,                    negative speed, absurd
    time-ordered)              20% overlap)                 distance/price dropped)
                                                                   │ clean records
                                                                   ▼
                                                            bin (median)
                                                                   │
                              ┌────────────────────────────────────┤
                              ▼                                    ▼
                     XGBoostUpdater                        ChronosUpdater
              finetune: xgb_model=prev booster        finetune: periodic LoRA step
              train:    fresh xgb.train               train:    short LoRA fit from base
                     │  (models_update/xgboost_stream)        │ (finetune/chronos_ft_stream)
                     └──────────────┬─────────────────────────┘
                                    ▼
                         PipelineWindowResult (DB)  ──poll──▶ Dashboard
```

Each layer has a single responsibility and a fixed interface; if you swap the
simulator for a real Kafka/HTTP consumer tomorrow, the lower layers don't change.

## Components

**StreamSimulator** (`streaming/simulator.py`) — "streams" the trip table in
time order as groups of `batch_size` (default 200) rows. It pages with a
`(time, id)` cursor and never loads the table into memory. The source is
selectable: `trip2017` (sefer_2017, fairly clean) or `train` (sefer_egitim,
dirty 2015-16 → the cleaner works harder, which shows the anomaly filter earning
its keep in the demo).

**SlidingWindow** (`streaming/window.py`) — keeps the last `W` (1000) raw records
in a ring buffer and emits a window every `S` (200) new records. Consecutive
windows overlap by `W−S`; each region enters the model more than once.

**IsolationForestCleaner** (`streaming/cleaning.py`) — fits a **fresh IF** each
time to the window's raw trip features (`trip_distance, trip_duration_minutes,
trip_speed_mph, total_amount, price_per_distance`) and drops the ones marked
`-1`. Why not the on-disk `models_update/if/*.pkl`: those score a specific
*binned series*; here we want an unsupervised outlier filter in the raw trip
space. Cleaning happens **before binning**, on the raw records → the underlying
median series stays clean.

**Updaters** (`streaming/updaters.py`) — take the clean records, bin them into a
median series at the target metric/resolution, and accumulate them in a rolling
series buffer:

- **XGBoostUpdater**
  - `train` → fresh `xgb.train` on the accumulated clean series (fresh booster).
  - `finetune` → `xgb.train(..., xgb_model=prev_booster)`: keeps the existing
    trees and boosts a few hundred more rounds on the new clean data. This is a
    **true incremental** update (init_model semantics).
  - Output: `models_update/xgboost_stream/*.pkl` (**exactly** the same package
    format as your training script: `features / lag_n / calibrator / model`).
    The panel reads this pkl from disk on every request, so an update is
    reflected immediately.

- **ChronosUpdater** — heavier, so it runs **every N windows** (default 5).
  - `train` → a short LoRA fit from the base `amazon/chronos-2`
    (`chronos_steps`, default 200 steps).
  - `finetune` → continues from the previous stream adapter with a **moderate
    LR** (periodic LoRA step). Note: adapter resume for Chronos-2 depends on the
    `fit()` version; the code passes `finetuned_ckpt` best-effort and falls back
    to a low-LR fit from base on versions that don't support it (noted in the
    docstring).
  - Output: `finetune/chronos_ft_stream/checkpoint`; after the fit, the merged
    pipeline is **hot-swapped** into `_MODEL_CACHE["chronos_stream"]` → the panel
    uses the updated model without a restart.

**Orchestrator** (`streaming/orchestrator.py`) — wires the loop: for each window,
clean → run the active updaters → write one `PipelineWindowResult` row per model →
print a summary to the console → call the `on_result` callback. `tick_sleep` sets
the demo pace and `max_windows` bounds it.

## Why separate "Live" model slots

The pipeline does **not** overwrite its trained/tuned baselines: it writes
XGBoost to `models_update/xgboost_stream/` and Chronos to
`finetune/chronos_ft_stream/`. Two new models were added to the panel:

| Panel model | Source | What it shows |
|---|---|---|
| **XGBoost (Live)** `xgboost_stream` | models_update/xgboost_stream | XGBoost updated incrementally from the stream |
| **Chronos-2 (Live)** `chronos_stream` | \_MODEL_CACHE / chronos_ft_stream | Chronos-2 periodically LoRA-tuned from the stream |

This lets you select **baseline vs. live** side by side in the panel — the "the
model updates itself as data flows" story in the demo comes straight from here.
Model + mode selection is done via the `run_pipeline` arguments (the "selection
part" you wanted).

## Django integration (5 steps)

1. Put the `streaming/` package inside the app: `anomaly/streaming/`.
2. Put `management/commands/run_pipeline.py` under
   `anomaly/management/commands/` (the management command must live at the app
   root, not under streaming).
3. Append the contents of `models_pipeline_snippet.py` to the end of
   `anomaly/models.py`, then
   `python manage.py makemigrations && python manage.py migrate`.
4. Take the updated `forecast_calculation.py` + `views.py` (XGBoost became
   family-parameterized; two Live models + their labels were added).
5. Run:

```bash
# XGBoost incremental + Chronos periodic, 50 windows from 2017, demo pace
python manage.py run_pipeline \
    --metric sefer --resolution 1h \
    --models xgboost,chronos --mode finetune \
    --source trip2017 --start "2017-01-01 00:00" \
    --window 1000 --step 200 --max-windows 50 --sleep 0.2

# XGBoost only, from-scratch training mode (train), from the dirty table (IF works harder)
python manage.py run_pipeline --models xgboost --mode train --source train --max-windows 30
```

On the console you'll see, per window:
```
[win  12] 2017-01-04 09:00  raw=1000 anom=23(2.3%) clean=977  |  xgboost_stream/finetune WAPE=4.81  chronos_stream/finetune:skip
```

## Showing it live on the dashboard

The results are in the `pipeline_window_result` table. A simple JSON endpoint is
enough to feed the panel (polled, e.g. every 2 s):

```python
# views.py
from django.http import JsonResponse
from .models import PipelineWindowResult

def pipeline_status(request):
    run = request.GET.get("run", "stream")
    qs = (PipelineWindowResult.objects
          .filter(run_name=run).order_by("-window_index", "model_name")[:60])
    return JsonResponse({"rows": list(qs.values(
        "window_index", "ts_start", "n_raw", "n_anomaly", "n_clean",
        "model_name", "mode", "status", "wape", "mae"))})
```

The front end takes this array and draws two things: a **WAPE-over-time chart**
(one line per model — the error dropping as updates progress) and **anomaly-rate**
bars. Alternatively, select the **XGBoost (Live)** / **Chronos-2 (Live)** models
in the forecast panel and watch the forecast curve directly — it updates as the
pipeline turns.

## Scaling (toward production)

- **Celery**: the body of `run_pipeline` is a single `run_pipeline(cfg)` call;
  wrapping it in a Celery task and triggering it with `beat` is enough. Add a
  broker (Redis) + worker and move the window processing into the task. For the
  demo the management command has fewer moving parts, so start with that.
- **Real stream**: replace `StreamSimulator` with a Kafka/HTTP consumer; the
  interface (an iterable producing batches) stays the same.
- **Chronos cost**: LoRA fit on CPU is slow; in the live loop keep
  `chronos_update_every` large or use a GPU. XGBoost is cheap every window.

## Limits / honest notes

- Chronos `finetune` true "adapter resume" depends on the library version; if
  unsupported, a short low-LR fit from base is done (it still improves with the
  accumulated clean series, but it isn't pure incremental). XGBoost `finetune`,
  by contrast, is full init_model.
- IF is fit fresh every window; `contamination` is fixed (default 2%). If the
  expected outlier rate varies a lot, making this adaptive is the next step.
- Warm-up: until the series buffer reaches `512+24` bins, Chronos returns
  "warmup"; XGBoost until `lag_n+60` bins. The first few windows pass without
  metrics — that's normal.
```
