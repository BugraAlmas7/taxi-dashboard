"""
streaming/config.py
Central config for the live streaming pipeline. One dataclass, no globals, so a
management command / Celery task / test can each build its own PipelineConfig.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineConfig:
    # ── what to forecast ─────────────────────────────────────────────────────
    # multi-metric: every retrain updates ALL of these (one live model per metric).
    # Only demand (trip count) + revenue (total_amount) are meaningful to forecast;
    # per-trip speed/distance/duration bin-averages are noisy and not actionable.
    metrics: List[str] = field(default_factory=lambda: ["sefer", "total_amount"])
    resolution: str = "1h"           # HOURLY bins (per the ops design)
    vendor: str = "hepsi"            # hepsi / "1" / "2"

    # retrain cadence: process a whole BLOCK of hours at once, then clean + retrain
    # (nightly batch). 24 = once per simulated day at midnight; 12 = twice a day.
    block_hours: int = 24
    # summary for value metrics: "mean" for the live pipeline (IF already removed
    # outliers, and mean is the natural expected-value target); "median" was only
    # needed against the 2015-16 raw pollution. 'sefer' is a count → unaffected.
    bin_how: str = "mean"

    # ── data source (the "test set" we stream from) ──────────────────────────
    #   "trip2017" → sefer_2017 (clean-ish panel data)
    #   "train"    → sefer_egitim (2015-16, outlier-polluted → IF works harder;
    #                good for showing the cleaner earn its keep in the demo)
    source: str = "trip2017"
    start: str = "2017-01-01 00:00"  # stream cursor start (YYYY-MM-DD HH:MM)
    batch_size: int = 2000           # raw trips pulled per simulator tick (bigger → pages a day faster)

    # (legacy sliding-window knobs — unused in the daily block_hours design)
    window_size: int = 1000
    window_step: int = 200

    # ── anomaly cleaning (Isolation Forest, raw-trip level) ──────────────────
    contamination: float = 0.026      # expected outlier fraction (or "auto")
    # IF cleans on TRIP characteristics only — total_amount is a forecast target,
    # so cleaning on it would be circular. (Must match train_common.CLEAN_FEATURES.)
    clean_features: List[str] = field(default_factory=lambda: [
        "trip_distance", "trip_duration_minutes", "trip_speed_mph", "price_per_distance",
    ])

    # ── which forecast models to update, and how ─────────────────────────────
    #   models: any of ["xgboost", "chronos"]
    #   mode  : "finetune" (incremental: xgb init_model / chronos LoRA step)
    #        or "train"    (fresh fit on the accumulated clean series)
    models: List[str] = field(default_factory=lambda: ["xgboost", "chronos"])
    mode: str = "finetune"

    # incremental-update knobs
    xgb_boost_rounds: int = 60       # trees added per finetune step (init_model)
    xgb_train_rounds: int = 300      # trees for a from-scratch train step
    chronos_update_every: int = 5    # run the (heavy) Chronos LoRA step every N windows
    chronos_steps: int = 200         # LoRA steps per Chronos update (keep small — live)

    # rolling clean-series buffer the updaters train on (bins, not raw rows)
    series_buffer_bins: int = 4000

    # ── run control ──────────────────────────────────────────────────────────
    max_windows: int = 0             # 0 = run until the stream is exhausted
    tick_sleep: float = 0.0          # seconds to pause per window (demo pacing)
    run_name: str = "stream"         # tag stored on every result row
    verbose: bool = True
