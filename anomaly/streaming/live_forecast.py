"""
streaming/live_forecast.py
PURE H-step-ahead forecasting for the live flowing panel.

Unlike the backtest path in forecast_calculation (_rolling_forecast re-anchors
with the ACTUAL values), the live forecast must NOT peek at the future — the
actuals ahead of "now" don't exist yet. So:
  • XGBoost family → recursive multi-step: predict step 1, feed it back as lag_1,
    predict step 2, ... using calendar features of each FUTURE bin timestamp.
  • Chronos  family → native multi-step from the context window (one shot).

Returns trailing actual + the forecast that extends PAST `now`, so the chart can
draw the forecast line leading the actual line.

Binning: MEAN by default (the live models are mean-trained; 'sefer' is a count so
mean/median is moot for it).
"""
from collections import defaultdict
import numpy as np

# Cache: (model_key, metric, resolution, vendor) -> { timestamp_str: forecast_value }
_LAGGED_FORECASTS = defaultdict(dict)
from datetime import timedelta

import numpy as np
import pandas as pd

from .. import time_series as ts
from ..forecast_calculation import (
    _binned_union, _resolution_seconds, _dt,
    _load_chronos_stream, _load_chronos_ft_optuna,
    _MODEL_CACHE  # include the cache mechanism
)
from .updaters import _lag_n, _time_feats

import os
import pickle

_PROJECT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODELS_UPDATE = os.path.join(_PROJECT, "models_update")
_FT_DIR = os.path.join(_PROJECT, "finetune")

# panel model key → (family kind, on-disk family / loader)
_XGB_FAMILY = {"xgboost_stream": "xgboost_stream", "xgboost": "xgboost"}
_CHRONOS_LOADER = {"chronos_stream": _load_chronos_stream,
                   "chronos_ft_optuna": _load_chronos_ft_optuna}


def _series_upto(metric, resolution, vendor, now_dt, n_back, how):
    dk = _resolution_seconds(resolution)
    lo = now_dt - timedelta(seconds=(n_back + 2) * dk)
    df = _binned_union(metric, resolution, vendor, lo, now_dt, how=how)
    if len(df) == 0:
        return df
    df["zaman"] = pd.to_datetime(df["zaman"])
    return df[df["zaman"] <= pd.Timestamp(now_dt)].tail(n_back).reset_index(drop=True)


def _future_times(now_dt, dk, horizon):
    return [pd.Timestamp(now_dt) + timedelta(seconds=dk * (i + 1)) for i in range(horizon)]


def _xgb_recursive(model_key, series_df, now_dt, resolution, horizon):
    """Recursive multi-step XGBoost forecast (no future actuals used)."""
    import xgboost as xgb
    family = _XGB_FAMILY.get(model_key, "xgboost")
    
    from .. import time_series as tsm
    metric = series_df.attrs.get("metric")
    vendor = series_df.attrs.get("vendor")
    
    pkl = os.path.join(_MODELS_UPDATE, family, f"xgboost_{metric}_{resolution}_{vendor}.pkl")
    
    # ---- FIXED FALLBACK ----
    # the leading "/" before "xgboost_" was removed entirely!
    if not os.path.exists(pkl) and family == "xgboost_stream":
        fallback_pkl = os.path.join(_MODELS_UPDATE, "xgboost", f"xgboost_{metric}_{resolution}_{vendor}.pkl")
        if os.path.exists(fallback_pkl):
            pkl = fallback_pkl
            
    if not os.path.exists(pkl):
        # instead of hiding the error, print the exact path we looked for
        return None, f"model not found: {pkl}"
        
    with open(pkl, "rb") as fh:
        pkg = pickle.load(fh)
        
    booster, feat_cols, n_lag = pkg["model"], pkg["features"], pkg["lag_n"]
    dk = _resolution_seconds(resolution)

    vals = series_df["deger"].to_numpy(dtype=float)
    if len(vals) < n_lag:
        return None, "not enough history"
        
    lags = list(vals[-n_lag:][::-1])            
    tfeat_names = [c for c in feat_cols if not c.startswith("lag_")]

    fc = []
    for ft in _future_times(now_dt, dk, horizon):
        row = {f"lag_{i+1}": lags[i] for i in range(n_lag)}
        tf = ts.add_time_features(pd.DataFrame({"zaman": [ft]}), "zaman").iloc[0]
        for name in tfeat_names:
            row[name] = float(tf[name])
        X = np.array([[row[c] for c in feat_cols]], dtype=np.float32)
        yhat = float(booster.predict(xgb.DMatrix(X))[0])
        fc.append(yhat)
        lags = [yhat] + lags[:-1]               
        
    return np.array(fc, dtype=float), None

def _chronos_ahead(model_key, series_df, resolution, horizon, context=512, metric=None):
    import torch
    
    # 1. Build the cache key and check memory
    cache_key = f"chronos_stream::{metric}" if model_key == "chronos_stream" else model_key
    
    if cache_key in _MODEL_CACHE:
        pipe, dev = _MODEL_CACHE[cache_key]
    else:
        # 2. Not cached: load from scratch and pull the weights from the correct path
        from chronos import BaseChronosPipeline
        from peft import PeftModel
        
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-2", device_map=dev,
            torch_dtype=torch.bfloat16 if dev == "cuda" else torch.float32
        )
        
        # path that matches the save logic in updaters.py exactly
        if model_key == "chronos_stream":
            ckpt_dir = os.path.join(_FT_DIR, "chronos_ft_stream", metric, "checkpoint")
        else:
            # Fallback for chronos_ft_optuna
            ckpt_dir = os.path.join(_FT_DIR, "chronos_ft_optuna", metric, "checkpoint")
            
        # if the weights folder exists, merge it (prevents the zero-shot fallback)
        if os.path.exists(ckpt_dir):
            pipe.model = PeftModel.from_pretrained(pipe.model, ckpt_dir).merge_and_unload()
            
        # cache it so the next tick avoids disk/network IO again
        _MODEL_CACHE[cache_key] = (pipe, dev)

    # 3. Standard forecasting logic
    ctx = series_df["deger"].to_numpy(dtype=np.float32)[-context:]
    if len(ctx) < 8:
        return None, "not enough context"
    
    x = torch.tensor(np.asarray(ctx, np.float32)[None, None, :], dtype=torch.float32)
    with torch.no_grad():
        qq, _ = pipe.predict_quantiles(x, prediction_length=horizon, quantile_levels=[0.5])
        
    if isinstance(qq, (list, tuple)):
        qq = np.stack([a.float().cpu().numpy() if hasattr(a, "cpu") else np.asarray(a, float)
                       for a in qq], 0)
    else:
        qq = qq.float().cpu().numpy() if hasattr(qq, "cpu") else np.asarray(qq, float)
        
    return np.asarray(qq, float).reshape(-1)[:horizon], None
def live_forecast(model_key, metric, resolution, vendor, now, horizon=60,
                  show_bins=180, context=512, how="mean"):
    """Return trailing actual + pure-ahead forecast for the flowing panel.

    now  : "YYYY-MM-DD HH:MM" (simulated current time)
    Returns dict: {now, actual:{x,y}, forecast:{x,y}, error?}.
    """
    now_dt = _dt(now) if isinstance(now, str) else now
    dk = _resolution_seconds(resolution)
    hist = _series_upto(metric, resolution, vendor, now_dt, context, how)
    if len(hist) == 0:
        return {"now": str(now_dt), "actual": {"x": [], "y": []},
                "forecast": {"x": [], "y": []}, "error": "no data"}
    hist.attrs["metric"] = metric
    hist.attrs["vendor"] = vendor

    try:
        if model_key in _CHRONOS_LOADER or model_key == "chronos_stream":
            fc, err = _chronos_ahead(model_key, hist, resolution, horizon, context, metric=metric)
        else:
            fc, err = _xgb_recursive(model_key, hist, now_dt, resolution, horizon)
    except Exception as e:
        msg = str(e)
        print(f"\n!!! FORECAST ERROR ({model_key}): {msg}\n")
        if "adapter_config.json" in msg or "Can't find" in msg or "No such file" in msg:
            err = f"'{model_key}' / {metric} model not trained yet — run the pipeline"
        else:
            err = f"{type(e).__name__}: {msg[:160]}"
        fc = None

    tail = hist.tail(show_bins)
    out = {
        "now": str(now_dt),
        "actual": {"x": [str(t) for t in tail["zaman"]],
                   "y": [float(v) for v in tail["deger"]]},
        "forecast": {"x": [], "y": []},
    }
    if err:
        out["error"] = err
        return out
    # ... [existing try/except forecast code] ...
    
    ft = _future_times(now_dt, dk, horizon)
    cache_key = (model_key, metric, resolution, vendor)
    
    # 1. Save our pure-ahead forecasts into the state cache
    if fc is not None:
        for t_obj, f_val in zip(ft, fc):
            _LAGGED_FORECASTS[cache_key][str(t_obj)] = float(f_val)

    # Trim old keys to avoid a potential memory leak
    if len(_LAGGED_FORECASTS[cache_key]) > 1000:
        keys_to_delete = sorted(_LAGGED_FORECASTS[cache_key].keys())[:-1000]
        for k in keys_to_delete:
            del _LAGGED_FORECASTS[cache_key][k]

    # 2. Match past forecasts against the actuals (tail) we now have
    y_true = []
    y_pred = []
    for t_val, a_val in zip(tail["zaman"], tail["deger"]):
        t_str = str(t_val)
        if t_str in _LAGGED_FORECASTS[cache_key]:
            y_true.append(float(a_val))
            y_pred.append(_LAGGED_FORECASTS[cache_key][t_str])

    # 3. Lagged WAPE computation
    lagged_wape = None
    if len(y_true) > 0:
        yt = np.array(y_true)
        yp = np.array(y_pred)
        denominator = np.sum(np.abs(yt))
        if denominator > 0:  # guard against division by zero
            lagged_wape = float(np.sum(np.abs(yt - yp)) / denominator)

    out = {
        "now": str(now_dt),
        "actual": {"x": [str(t) for t in tail["zaman"]],
                   "y": [float(v) for v in tail["deger"]]},
        "forecast": {"x": [], "y": []},
        "lagged_wape": lagged_wape  # send the already-computed WAPE straight to the frontend
    }
    
    if err:
        out["error"] = err
        return out
        
    out["forecast"] = {
        "x": [str(tail["zaman"].iloc[-1])] + [str(t) for t in ft],
        "y": [float(tail["deger"].iloc[-1])] + [float(v) for v in fc],
    }
    return out
