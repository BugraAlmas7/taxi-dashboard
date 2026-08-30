"""
streaming/updaters.py
Model updaters fed the CLEANED window each step. Two modes:

  • "train"    — fit a fresh model on the accumulated clean series.
  • "finetune" — incremental:
        XGBoost → xgb.train(..., xgb_model=prev_booster): keeps the existing
                  trees and boosts a few more rounds on the new clean data.
        Chronos → a periodic short LoRA step on the growing clean buffer.

The pipeline writes to SEPARATE "live" model slots so your trained/tuned
baselines stay intact and the panel can show baseline vs live side by side:
    XGBoost → models_update/xgboost_stream/*.pkl   (panel model "xgboost_stream")
    Chronos → finetune/chronos_ft_stream/checkpoint (panel model "chronos_stream")
"""
import os
import pickle

import numpy as np
import pandas as pd

from .. import time_series as ts
from ..forecast_calculation import _metrics, _MODEL_CACHE
from . import params as live_params

# on-disk targets (kept out of the baseline folders on purpose)
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # <app>/..
_MODELS_UPDATE = os.path.join(_PROJECT, "models_update")
_FT_DIR = os.path.join(_PROJECT, "finetune")


# ── shared: a rolling clean binned series the updaters train on ──────────────
class SeriesBuffer:
    """Accumulates the cleaned, binned series across windows (dedup by bin ts,
    newest value wins), capped at `max_bins`."""
    def __init__(self, metric, resolution, max_bins, how="mean"):
        self.metric, self.resolution, self.max_bins = metric, resolution, max_bins
        self.how = how
        self._by_ts = {}                      # zaman -> deger

    def add_clean(self, clean_df):
        binned = ts.bin_rows(clean_df, self.metric, ts.resolution_freq(self.resolution),
                             how=self.how)
        for z, v in zip(binned["zaman"], binned["deger"]):
            self._by_ts[pd.Timestamp(z)] = float(v)
        if len(self._by_ts) > self.max_bins:          # trim oldest
            for z in sorted(self._by_ts)[: len(self._by_ts) - self.max_bins]:
                del self._by_ts[z]
        return binned

    def frame(self):
        if not self._by_ts:
            return pd.DataFrame(columns=["zaman", "deger"])
        z = sorted(self._by_ts)
        return pd.DataFrame({"zaman": z, "deger": [self._by_ts[t] for t in z]})


# ── feature builder (mirrors training_forecast_models.py) ────────────────────
def _lag_n(dk):    return 60 if dk < 3600 else (48 if dk < 86400 else 14)
def _time_feats(dk):
    if dk >= 86400: return ["wd"]
    if dk >= 3600:  return ["wd", "saat"]
    return ["wd", "saat", "dakika"]


def _build_xy(frame, dk, n_lag, feat_cols):
    """From a contiguous [zaman,deger] frame build (X, y) with lag_1..n + time
    feats, exactly the column order compute_xgboost expects."""
    f = ts.add_time_features(frame.sort_values("zaman").reset_index(drop=True), "zaman")
    y = f["deger"].to_numpy(dtype=np.float32)
    for lag in range(1, n_lag + 1):
        f[f"lag_{lag}"] = f["deger"].shift(lag)
    f = f.dropna(subset=feat_cols)
    X = f[feat_cols].to_numpy(dtype=np.float32)
    yv = f["deger"].to_numpy(dtype=np.float32)
    return X, yv


def _calibrator(resid):
    return {"width_90": float(np.quantile(resid, 0.90)),
            "width_95": float(np.quantile(resid, 0.95))}


# ── XGBoost updater (init_model incremental / fresh train) ───────────────────
class XGBoostUpdater:
    family = "xgboost_stream"
    label  = "xgboost_stream"

    def __init__(self, cfg, metric):
        self.cfg = cfg
        self.metric = metric
        self.dk = _resolution_seconds(cfg.resolution)
        self.buf = SeriesBuffer(metric, cfg.resolution, cfg.series_buffer_bins, cfg.bin_how)
        self._pkl = os.path.join(_MODELS_UPDATE, self.family,
                                 f"xgboost_{metric}_{cfg.resolution}_{cfg.vendor}.pkl")
        # Optuna-tuned XGBoost params (live_params.json) if present, else defaults
        self.xgb_kwargs, self.tuned_rounds, tuned_lag = live_params.xgb_params(metric)
        self.n_lag = tuned_lag or _lag_n(self.dk)

    def update(self, clean_df, window_index):
        import xgboost as xgb
        self.buf.add_clean(clean_df)
        frame = self.buf.frame()

        n_lag = self.n_lag
        feat_cols = [f"lag_{i}" for i in range(1, n_lag + 1)] + _time_feats(self.dk)
        if len(frame) < n_lag + 60:
            return {"model": self.label, "metric": self.metric, "mode": self.cfg.mode,
                    "status": "warmup", "n_train": len(frame)}

        X, y = _build_xy(frame, self.dk, n_lag, feat_cols)
        if len(X) < 40:
            return {"model": self.label, "metric": self.metric, "mode": self.cfg.mode,
                    "status": "warmup", "n_train": len(X)}

        # time-ordered holdout tail for telemetry
        cut = max(1, int(len(X) * 0.9))
        Xtr, ytr, Xte, yte = X[:cut], y[:cut], X[cut:], y[cut:]

        params = {"objective": "reg:squarederror", "eval_metric": "rmse",
                  "max_bin": 127, "tree_method": "hist", **self.xgb_kwargs}
        dtr = xgb.DMatrix(Xtr, label=ytr)

        prev = self._load_prev() if self.cfg.mode == "finetune" else None
        # fresh train → tuned round count; incremental step → cfg.xgb_boost_rounds
        rounds = self.cfg.xgb_boost_rounds if prev is not None else self.tuned_rounds
        booster = xgb.train(params, dtr, num_boost_round=rounds, xgb_model=prev)

        # metrics on the holdout tail
        wape = mae = None
        if len(Xte):
            pred = booster.predict(xgb.DMatrix(Xte))
            mt = _metrics(yte, pred)
            if mt: wape, mae = mt["wape"], mt["mae"]
            resid = np.abs(yte - pred)
        else:
            resid = np.abs(ytr - booster.predict(xgb.DMatrix(Xtr)))

        pkg = {"kind": "xgb", "model": booster, "calibrator": _calibrator(resid),
               "features": feat_cols, "lag_n": n_lag,
               "coz": self.cfg.resolution, "metrik": self.metric, "vendor": self.cfg.vendor}
        os.makedirs(os.path.dirname(self._pkl), exist_ok=True)
        with open(self._pkl, "wb") as fh:
            pickle.dump(pkg, fh)
        # compute_xgboost_stream reads this pkl from disk per request → no cache to bust

        return {"model": self.label, "metric": self.metric, "mode": self.cfg.mode, "status": "ok",
                "wape": wape, "mae": mae, "n_train": len(Xtr),
                "trees": booster.num_boosted_rounds(),
                "incremental": prev is not None}

    def _load_prev(self):
        if not os.path.exists(self._pkl):
            return None
        try:
            with open(self._pkl, "rb") as fh:
                return pickle.load(fh)["model"]
        except Exception:
            return None


# ── Chronos-2 updater (periodic LoRA step) ───────────────────────────────────
class ChronosUpdater:
    label = "chronos_stream"

    def __init__(self, cfg, metric):
        self.cfg = cfg
        self.metric = metric
        self.buf = SeriesBuffer(metric, cfg.resolution, cfg.series_buffer_bins, cfg.bin_how)
        # per-metric adapter dir so metrics don't clobber each other's checkpoint
        self.ckpt_dir = os.path.join(_FT_DIR, "chronos_ft_stream", metric)
        self.cache_key = f"chronos_stream::{metric}"
        self._last = {"model": self.label, "metric": metric, "mode": cfg.mode, "status": "warmup"}

    def update(self, clean_df, window_index):
        self.buf.add_clean(clean_df)
        # heavy → only every N windows
        if (window_index + 1) % self.cfg.chronos_update_every != 0:
            return {**self._last, "status": "skip"}

        series = self.buf.frame()["deger"].to_numpy(dtype=np.float32)
        if len(series) < 512 + 24:
            self._last = {"model": self.label, "metric": self.metric, "mode": self.cfg.mode,
                          "status": "warmup", "n_train": int(len(series))}
            return self._last
        try:
            import torch
            from chronos import BaseChronosPipeline
            from peft import PeftModel
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            os.makedirs(self.ckpt_dir, exist_ok=True)

            base = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-2", device_map=dev,
                torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
            targets = ["self_attention.q", "self_attention.v", "self_attention.k",
                       "self_attention.o", "output_patch_embedding.output_layer"]
            # finetune → warm-start from the previous stream adapter if present
            prev = os.path.join(self.ckpt_dir, "checkpoint")
            warm = self.cfg.mode == "finetune" and os.path.exists(prev)
            lr = 3.87e-4 * (0.3 if warm else 1.0)     # gentler LR when continuing

            fit_kwargs = dict(
                prediction_length=24, finetune_mode="lora",
                lora_config={"r": 8, "lora_alpha": 64, "target_modules": targets},
                context_length=512, learning_rate=lr, num_steps=self.cfg.chronos_steps,
                batch_size=16, output_dir=self.ckpt_dir,
                finetuned_ckpt_name="checkpoint", disable_tqdm=True)
            # some chronos versions accept a resume/init ckpt; pass it best-effort
            if warm:
                fit_kwargs["finetuned_ckpt"] = prev
            try:
                ft = base.fit([{"target": series}], **fit_kwargs)
            except TypeError:
                fit_kwargs.pop("finetuned_ckpt", None)   # older signature
                ft = base.fit([{"target": series}], **fit_kwargs)

            # hot-swap the panel's live model: merge adapter into a fresh pipe
            pipe = BaseChronosPipeline.from_pretrained(
                "amazon/chronos-2", device_map=dev,
                torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32)
            pipe.model = PeftModel.from_pretrained(pipe.model, prev).merge_and_unload()
            _MODEL_CACHE[self.cache_key] = (pipe, dev)

            wape = self._holdout_wape(pipe, series)
            self._last = {"model": self.label, "metric": self.metric, "mode": self.cfg.mode,
                          "status": "ok", "wape": wape, "n_train": int(len(series)),
                          "incremental": warm}
        except Exception as e:
            self._last = {"model": self.label, "metric": self.metric, "mode": self.cfg.mode,
                          "status": "error", "error": f"{type(e).__name__}: {e}"}
        return self._last

    def _holdout_wape(self, pipe, series, C=512, H=24):
        import torch
        if len(series) < C + H:
            return None
        ctx, tgt = series[-(C + H):-H], series[-H:]
        x = torch.tensor(np.asarray(ctx, np.float32)[None, None, :], dtype=torch.float32)
        with torch.no_grad():
            qq, _ = pipe.predict_quantiles(x, prediction_length=H, quantile_levels=[0.5])
        if isinstance(qq, (list, tuple)):
            qq = np.stack([a.float().cpu().numpy() if hasattr(a, "cpu") else np.asarray(a, float)
                           for a in qq], 0)
        else:
            qq = qq.float().cpu().numpy() if hasattr(qq, "cpu") else np.asarray(qq, float)
        pred = np.asarray(qq, float).reshape(-1)[:H]
        mt = _metrics(tgt, pred)
        return mt["wape"] if mt else None


# local copy (avoid importing the private name if it moves)
def _resolution_seconds(resolution):
    table = {"1m": 60, "3m": 180, "5m": 300, "10m": 600, "15m": 900, "30m": 1800,
             "1h": 3600, "3h": 10800, "1d": 86400}
    return table.get(resolution, 3600)


UPDATERS = {"xgboost": XGBoostUpdater, "chronos": ChronosUpdater}
