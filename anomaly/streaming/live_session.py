"""
streaming/live_session.py
Server-side state for the DISPLAY-DRIVEN nightly retrain. The Live page advances
hourly; when it crosses midnight it calls retrain(day) which cleans that day's
raw trips and updates the live model(s) — synchronously, so the page can pause
and wait for it to finish, then resume.

Holds one set of per-metric updaters + their rolling buffers, pre-warmed with the
history BEFORE the session start (so the very first forecast is already trained,
and nightly retrains only append one day).
"""
import time
from datetime import datetime, timedelta

import pandas as pd

from .config import PipelineConfig
from .cleaning import IsolationForestCleaner
from .updaters import UPDATERS
from ..models import Trip2017, TrainTrip

_SOURCES = {"trip2017": Trip2017, "train": TrainTrip}


class _LiveSession:
    def __init__(self):
        self.key = None
        self.cfg = None
        self.cleaner = None
        self.updaters = []
        self.block_index = 0

    def _cols(self):
        # clean features (for IF) + total_amount (a target, not a clean feature) + vendorid
        cols = ["tpep_pickup_datetime"] + list(self.cfg.clean_features) + ["total_amount", "vendorid"]
        return list(dict.fromkeys(cols))

    def _pull(self, source, lo, hi, vendor):
        model = _SOURCES.get(source, Trip2017)
        qs = model.objects.filter(tpep_pickup_datetime__range=(lo, hi))
        if vendor != "hepsi":
            qs = qs.filter(vendorid=vendor)
        df = pd.DataFrame(list(qs.values(*self._cols())))
        if len(df):
            df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        return df

    def _ensure(self, metrics, model, mode, resolution, vendor, source, start,
                prewarm_days=30, train_source="train"):
        key = (tuple(metrics), model, mode, resolution, vendor, source, start)
        if self.key == key:
            return
        self.cfg = PipelineConfig(metrics=list(metrics), models=[model], mode=mode,
                                  resolution=resolution, vendor=vendor, source=source)
        self.cleaner = IsolationForestCleaner(self.cfg)
        self.updaters = [UPDATERS[model](self.cfg, m) for m in metrics]
        self.block_index = 0
        # pre-warm from the TRAINING source (2015-16) tail — NOT the 2017 test data,
        # so the model never sees the demo period. Pull DAY-BY-DAY (one day ≈ a few
        # hundred k rows) to keep RAM bounded — 30 days at once would be ~10M rows → OOM.
        start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M")
        total = 0
        for d in range(prewarm_days, 0, -1):
            lo = start_dt - timedelta(days=d)
            hi = start_dt - timedelta(days=d - 1)
            raw = self._pull(train_source, lo, hi, vendor)
            if len(raw) == 0:
                continue
            clean, _ = self.cleaner.clean(raw)
            for up in self.updaters:
                up.buf.add_clean(clean)          # accumulate only; train once below
            total += len(raw)
            del raw, clean
        # single training pass over the accumulated prewarm buffer
        empty = pd.DataFrame(columns=self._cols())
        for up in self.updaters:
            try:
                up.update(empty, self.block_index)
            except Exception as e:
                print(f"[live_session] prewarm train {getattr(up,'metric','?')}: {e}")
        self.block_index += 1
        self.key = key
        print(f"[live_session] ready · {model}/{mode} · prewarm {total} rows (day-by-day)")

    def retrain(self, day, metrics, model, mode, resolution, vendor, source, start):
        """Clean the given day's raw trips and update the live model(s). Returns
        timing + cleaning stats + per-model outcome."""
        self._ensure(metrics, model, mode, resolution, vendor, source, start)
        lo = datetime.strptime(day, "%Y-%m-%d")
        hi = lo + timedelta(days=1)
        raw = self._pull(source, lo, hi, vendor)
        if len(raw) == 0:
            return {"day": day, "n_raw": 0, "n_anomaly": 0, "n_clean": 0,
                    "anom_rate": 0.0, "took": 0.0, "models": []}
        clean, stats = self.cleaner.clean(raw)
        t0 = time.time()
        per = []
        for up in self.updaters:
            try:
                per.append(up.update(clean, self.block_index))
            except Exception as e:
                per.append({"metric": getattr(up, "metric", "?"), "status": "error",
                            "error": f"{type(e).__name__}: {e}"})
        self.block_index += 1
        return {"day": day, **stats, "took": round(time.time() - t0, 2), "models": per}


SESSION = _LiveSession()
