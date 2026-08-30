"""
forecast_calculation.py
One function per forecast model + a MODELS dispatch dict.
Signature matches anomaliy_calculation.

Data access: the ORM only pulls raw rows; union/binning/summary are done entirely
in pandas. No SQL anywhere. The two tables (training + 2017) are pulled raw and
concatenated BEFORE binning, so the summary is exact everywhere (no boundary-bin
approximation).

Metric summary = MEDIAN — this MUST match training. training_forecast_models.py
bins with a Postgres PERCENTILE_CONT (median) aggregate, so the lag features the
models learned from are medians. Median is robust to the outliers in the 2015-16
training table (negative / thousands-of-mph speeds); mean binning let that
garbage into the targets and the trees learned nonsense. After changing this you
MUST re-run training_forecast_models.py so the .pkl files are median-trained too.
"""
import os, pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .models import Trip2017, TrainTrip
from . import time_series as ts
from . import calibration as cal_mod

_CFG = {"scrollZoom": True}


def _resolution_seconds(resolution):
    table = {
        "1s":1,    "3s":3,    "5s":5,    "10s":10,  "15s":15,  "30s":30,
        "1m":60,   "3m":180,  "5m":300,  "10m":600, "15m":900, "30m":1800,
        "1h":3600, "3h":10800,"1d":86400,
    }
    return table.get(resolution, 3600)


def _dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


# ── Pull + concat + bin (two tables, all pandas) ─────────────────────────────

def _raw_two_tables(metric, vendor, start_dt, end_dt):
    """Pull raw rows from both trip tables and concat into ONE DataFrame."""
    columns = ["tpep_pickup_datetime"] + ([] if metric == "sefer" else [metric])
    parts = []
    for model in (TrainTrip, Trip2017):
        qs = model.objects.filter(tpep_pickup_datetime__range=(start_dt, end_dt))
        if vendor != "hepsi":
            qs = qs.filter(vendorid=vendor)      # vendorid is text → compare as string
        parts.append(pd.DataFrame(list(qs.values(*columns))))
    parts = [p for p in parts if len(p)]
    if not parts:
        return pd.DataFrame(columns=columns)
    return pd.concat(parts, ignore_index=True)


def _binned_union(metric, resolution, vendor, start_dt, end_dt, how="median"):
    """Bin the concatenated raw rows in pandas.

    Summary = MEDIAN, to match training_forecast_models.py which now bins with a
    Postgres PERCENTILE_CONT (median) aggregate. Median is robust to the outliers
    polluting the 2015-16 training table (negative / thousands-of-mph speeds), so
    both train and serve see clean values. (NOTE: requires the models to be
    RE-TRAINED with the median training script — mean-trained .pkl + median serve
    is a mismatch.)
    """
    raw = _raw_two_tables(metric, vendor, start_dt, end_dt)
    return ts.bin_rows(raw, metric, ts.resolution_freq(resolution), how=how)


# ── Shared helpers for the foundation models (Chronos / TimesFM / TimeGPT) ───

_MODEL_CACHE = {}


def _context_window(metric, resolution, vendor, start, end, max_ctx=1024, how="median"):
    """
    Context = last max_ctx bins BEFORE start (training + 2017 → reaches the past).
    Horizon = the WHOLE [start, end] window (actual values).
    Returns: (ctx_df[zaman,deger], win_df[zaman,deger]) or (None, None).
    """
    dk       = _resolution_seconds(resolution)
    start_dt = _dt(start)
    end_dt   = _dt(end)
    buf_lo   = start_dt - timedelta(seconds=max_ctx * dk)

    df = _binned_union(metric, resolution, vendor, buf_lo, end_dt, how=how)
    if len(df) == 0:
        return None, None
    df["zaman"] = pd.to_datetime(df["zaman"])

    ctx_df = df[df["zaman"] < pd.Timestamp(start_dt)].tail(max_ctx).reset_index(drop=True)
    win_df = df[df["zaman"] >= pd.Timestamp(start_dt)].reset_index(drop=True)
    if len(ctx_df) < 8 or len(win_df) < 1:
        return None, None
    return ctx_df[["zaman", "deger"]], win_df[["zaman", "deger"]]


def _calib_q(level):
    return {"95": (0.025, 0.975), "90": (0.05, 0.95), "yok": None}.get(level, (0.025, 0.975))


_PD_FREQ = {
    "1m":"min","3m":"3min","5m":"5min","10m":"10min","15m":"15min","30m":"30min",
    "1h":"h","3h":"3h","1d":"D",
}

_SEASON = {
    "1s":3600, "3s":1200, "5s":720, "10s":360, "15s":240, "30s":120,
    "1m":1440, "3m":480,  "5m":288, "10m":144, "15m":96,  "30m":48,
    "1h":24,   "3h":8,    "1d":7,
}


def _metrics(y_true, y_pred):
    """WAPE / MAPE / MAE / RMSE / MASE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = min(len(y_true), len(y_pred))
    y_true, y_pred = y_true[:n], y_pred[:n]
    msk = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[msk], y_pred[msk]
    if len(y_true) == 0:
        return None
    err  = np.abs(y_true - y_pred)
    mae  = float(err.mean())
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.abs(y_true).sum()
    wape = float(err.sum() / denom * 100) if denom else float("nan")
    nz = y_true != 0
    mape = float(np.mean(err[nz] / np.abs(y_true[nz])) * 100) if nz.any() else float("nan")
    naive = np.abs(np.diff(y_true)).mean()
    mase = float(mae / naive) if naive else float("nan")
    return {"n": int(len(y_true)), "wape": wape, "mape": mape, "mae": mae, "rmse": rmse, "mase": mase}


def _metrics_label(mt):
    if mt is None:
        return "no metrics (no aligned actual values found)"
    mape = "—" if np.isnan(mt["mape"]) else f"{mt['mape']:.1f}%"
    return (f"WAPE {mt['wape']:.1f}% · MAPE {mape} · "
            f"MAE {mt['mae']:.2f} · RMSE {mt['rmse']:.2f} · n={mt['n']} · MASE {mt['mase']:.2f}")


def _forecast_chart(win_df, y_med, y_lo, y_hi, title, color, level, resolution,
                    start_date, start_time, end_date, end_time):
    """Shared forecast chart."""
    x_win = win_df["zaman"]
    x_fc  = win_df["zaman"].iloc[:len(y_med)]
    fig = go.Figure()

    lo_arr = hi_arr = None
    if y_lo is not None and y_hi is not None:
        lo_arr = np.maximum(np.asarray(y_lo, dtype=float), 0.0)[:len(x_fc)]
        hi_arr = np.asarray(y_hi, dtype=float)[:len(x_fc)]
        a = 0.20 if level == "90" else 0.12
        mean_w = float(np.mean((hi_arr - lo_arr) / 2))
        fig.add_scatter(
            x=pd.concat([x_fc, x_fc[::-1]]),
            y=np.concatenate([hi_arr, lo_arr[::-1]]),
            fill="toself", fillcolor=f"rgba(28,114,147,{a})",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{level}% band (avg ±{mean_w:.2f})", showlegend=True
        )

    fig.add_scatter(x=x_fc, y=y_med, mode="lines",
                    name=f"{title} forecast", line=dict(color=color, width=2))
    fig.add_scatter(x=x_win, y=win_df["deger"], mode="lines",
                    name="actual", line=dict(color="black", width=1.5))

    if lo_arr is not None:
        g_arr = win_df["deger"].to_numpy(dtype=float)[:len(x_fc)]
        outside = (g_arr < lo_arr) | (g_arr > hi_arr)
        if outside.any():
            fig.add_scatter(
                x=x_fc.to_numpy()[outside], y=g_arr[outside],
                mode="markers", name=f"out of band ({int(outside.sum())})",
                marker=dict(color="#c0392b", size=8, symbol="x",
                            line=dict(width=1, color="#c0392b"))
            )

    mt = _metrics(win_df["deger"].to_numpy()[:len(y_med)], np.asarray(y_med))
    fig.update_layout(
        title=f"{title} · {resolution} · {start_date} {start_time}→{end_date} {end_time}"
              f"<br><sub>{_metrics_label(mt)}</sub>",
        height=480, template="plotly_white", legend=dict(orientation="h", y=-0.15)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_CFG)


def compute_lgbm(metric, resolution, vendor, start, end, _vf, period, k,
                 start_date, start_time, end_date, end_time, level="95"):
    """
    LightGBM point forecast + conformal confidence band (level: yok/90/95).
    Binning consistent with training (pandas floor); metric summary = MEDIAN.
    Time features come from the bin timestamp (Sunday=0 convention).
    Buffer: n_lag*dk seconds before start (training+2017 union) → no warm-up gap.
    """
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_update")
    pkl_path = ts.model_path(MODEL_DIR, "lgbm", metric, resolution, vendor)
    if not os.path.exists(pkl_path):
        return (f"<p style='color:#c0392b'>Model not found: {os.path.basename(pkl_path)} "
                f"— run training_forecast_models.py.</p>")

    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    model     = pkg["model"]
    feat_cols = pkg["features"]
    n_lag     = pkg["lag_n"]
    cal       = pkg["calibrator"]

    dk       = _resolution_seconds(resolution)
    start_dt = _dt(start)
    end_dt   = _dt(end)
    buf_lo   = start_dt - timedelta(seconds=n_lag * dk)

    buf = _binned_union(metric, resolution, vendor, buf_lo, end_dt)
    if len(buf) == 0:
        return "<p style='color:#c0392b'>No data in this range</p>"

    buf = ts.add_time_features(buf, "zaman")     # wd/saat/dakika/saniye
    buf = buf.sort_values("zaman").reset_index(drop=True)

    for lag in range(1, n_lag + 1):
        buf[f"lag_{lag}"] = buf["deger"].shift(lag)

    inf = buf[buf["zaman"] >= pd.Timestamp(start_dt)].dropna(subset=feat_cols).copy()
    if len(inf) == 0:
        return "<p style='color:#c0392b'>Not enough preceding data for lag features</p>"

    X = inf[feat_cols].values.astype(float)
    inf["tahmin"] = model.predict(X)

    fig = go.Figure()
    ab = cal_mod.band(inf["tahmin"].values, cal, level)
    lower = upper = None
    if ab is not None:
        lower, upper, w = ab
        fig.add_scatter(
            x=pd.concat([inf["zaman"], inf["zaman"][::-1]]),
            y=pd.concat([pd.Series(upper), pd.Series(lower)[::-1]]),
            fill="toself", fillcolor=cal_mod.fill_color(level),
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{cal_mod.label(level)} (±{w:.2f})", showlegend=True
        )

    fig.add_scatter(x=inf["zaman"], y=inf["tahmin"], mode="lines",
                    name="LightGBM forecast", line=dict(color="#1C7293", width=2))

    actual = buf[buf["zaman"] >= pd.Timestamp(start_dt)].copy()
    fig.add_scatter(x=actual["zaman"], y=actual["deger"], mode="lines",
                    name="actual", line=dict(color="black", width=1.5))

    if lower is not None:
        g_arr = inf["deger"].to_numpy(dtype=float)
        outside = (g_arr < np.asarray(lower)) | (g_arr > np.asarray(upper))
        if outside.any():
            fig.add_scatter(
                x=inf["zaman"].to_numpy()[outside], y=g_arr[outside],
                mode="markers", name=f"out of band ({int(outside.sum())})",
                marker=dict(color="#c0392b", size=8, symbol="x",
                            line=dict(width=1, color="#c0392b"))
            )

    mt = _metrics(inf["deger"].to_numpy(), inf["tahmin"].to_numpy())
    fig.update_layout(
        title=f"{metric} · LightGBM · {resolution} · "
              f"{start_date} {start_time}→{end_date} {end_time}"
              f"<br><sub>{_metrics_label(mt)}</sub>",
        height=480, template="plotly_white", legend=dict(orientation="h", y=-0.15)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_CFG)


def compute_chronos(metric, resolution, vendor, start, end, _vf, period, k,
                    start_date, start_time, end_date, end_time, level="95"):
    """Chronos (Amazon) zero-shot — with a ROLLING window."""
    try:
        import torch
        from chronos import ChronosPipeline
    except ImportError:
        return ("<p style='color:#c0392b'>Chronos not installed: "
                "<code>pip install chronos-forecasting</code> (needs torch).</p>")

    ctx_df, win_df = _context_window(metric, resolution, vendor, start, end, max_ctx=1024)
    if ctx_df is None:
        return "<p style='color:#c0392b'>Not enough history/window data</p>"

    STEP    = 64
    MAX_CTX = 1024
    try:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if "chronos" not in _MODEL_CACHE:
            _MODEL_CACHE["chronos"] = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-large", device_map=dev, torch_dtype=torch.float32)
        pipe = _MODEL_CACHE["chronos"]

        q = _calib_q(level)
        actual_win = win_df["deger"].to_numpy(dtype="float32")
        history = list(ctx_df["deger"].to_numpy(dtype="float32"))

        H = len(win_df)
        y_med = np.empty(H, dtype=float)
        y_lo  = np.empty(H, dtype=float) if q else None
        y_hi  = np.empty(H, dtype=float) if q else None

        i = 0
        while i < H:
            h = int(min(STEP, H - i))
            ctx_t = torch.tensor(np.asarray(history[-MAX_CTX:], dtype="float32"),
                                 dtype=torch.float32)
            fc  = pipe.predict(ctx_t, h)
            arr = fc[0].cpu().numpy()
            y_med[i:i+h] = np.quantile(arr, 0.5, axis=0)
            if q:
                y_lo[i:i+h] = np.quantile(arr, q[0], axis=0)
                y_hi[i:i+h] = np.quantile(arr, q[1], axis=0)
            history.extend(actual_win[i:i+h].tolist())     # re-anchor with actuals
            i += h
    except Exception as e:
        return f"<p style='color:#c0392b'>Chronos error: {type(e).__name__}: {e}</p>"

    return _forecast_chart(win_df, y_med, y_lo, y_hi, "Chronos", "#7D3C98", level,
                           resolution, start_date, start_time, end_date, end_time)


def compute_timesfm(metric, resolution, vendor, start, end, _vf, period, k,
                    start_date, start_time, end_date, end_time, level="95"):
    """TimesFM (Google) zero-shot."""
    try:
        import timesfm
    except ImportError:
        return ("<p style='color:#c0392b'>TimesFM not installed: "
                "<code>pip install timesfm[torch]</code></p>")

    ctx_df, win_df = _context_window(metric, resolution, vendor, start, end, max_ctx=1024)
    if ctx_df is None:
        return "<p style='color:#c0392b'>Not enough history/window data</p>"
    ctx = len(ctx_df)
    H   = len(win_df)

    try:
        import torch
        dev = "gpu" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"

    def _api_info():
        attrs = [a for a in dir(timesfm) if not a.startswith("_")]
        return f"timesfm {getattr(timesfm, '__version__', '?')} → {attrs}"

    MAX_CTX, MAX_H = 1024, 256
    try:
        if "timesfm" not in _MODEL_CACHE:
            if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
                cls  = timesfm.TimesFM_2p5_200M_torch
                repo = "google/timesfm-2.5-200m-pytorch"
                if hasattr(cls, "from_pretrained"):
                    m = cls.from_pretrained(repo)
                else:
                    m = cls()
                    m.load_checkpoint(repo)
                m.compile(timesfm.ForecastConfig(
                    max_context=MAX_CTX, max_horizon=MAX_H, normalize_inputs=True))
                _MODEL_CACHE["timesfm"] = ("2p5", m)
            elif hasattr(timesfm, "TimesFm") and hasattr(timesfm, "TimesFmHparams"):
                ctx_len = min(512, max(32, (ctx // 32) * 32))
                m = timesfm.TimesFm(
                    hparams=timesfm.TimesFmHparams(
                        backend=dev, per_core_batch_size=32,
                        horizon_len=min(H, MAX_H), context_len=ctx_len,
                        num_layers=20, model_dims=1280,
                        input_patch_len=32, output_patch_len=128),
                    checkpoint=timesfm.TimesFmCheckpoint(
                        huggingface_repo_id="google/timesfm-1.0-200m-pytorch"))
                _MODEL_CACHE["timesfm"] = ("1p2", m)
            elif hasattr(timesfm, "TimesFm"):
                ctx_len = min(512, max(32, (ctx // 32) * 32))
                m = timesfm.TimesFm(
                    context_len=ctx_len, horizon_len=min(H, MAX_H),
                    input_patch_len=32, output_patch_len=128,
                    num_layers=20, model_dims=1280, backend=dev)
                m.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")
                _MODEL_CACHE["timesfm"] = ("1p0", m)
            else:
                return f"<p style='color:#c0392b'>TimesFM API not recognized.<br>{_api_info()}</p>"

        tag, m = _MODEL_CACHE["timesfm"]
        Hc = int(min(len(win_df), MAX_H))
        ctx_use = ctx_df["deger"].to_numpy(dtype="float32")[-MAX_CTX:]

        quant = None
        if tag == "2p5":
            point, quant = m.forecast(horizon=Hc, inputs=[ctx_use])
        else:
            res = m.forecast([ctx_use], freq=[0])
            point = res[0]
            quant = res[1] if len(res) > 1 else None
        y_med = np.asarray(point[0])[:Hc]

        y_lo = y_hi = None
        if level != "yok" and quant is not None:
            q = np.asarray(quant[0])
            if q.ndim == 2 and q.shape[1] >= 2:
                y_lo = q.min(axis=1)[:Hc]
                y_hi = q.max(axis=1)[:Hc]
    except Exception as e:
        return (f"<p style='color:#c0392b'>TimesFM error: {type(e).__name__}: {e}<br>"
                f"{_api_info()}</p>")

    return _forecast_chart(win_df, y_med, y_lo, y_hi, "TimesFM", "#1E8449", level,
                           resolution, start_date, start_time, end_date, end_time)


def compute_xgboost(metric, resolution, vendor, start, end, _vf, period, k,
                 start_date, start_time, end_date, end_time, level="95",
                 _family="xgboost", _title="XGBoost", _color="#D35400", _how="median"):
    """XGBoost point forecast + conformal confidence band.

    _family/_title/_color let the streaming 'live' variant reuse this whole body
    while reading its own pkl folder (models_update/xgboost_stream); _how switches
    the serving summary (mean for the mean-trained live model)."""
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_update")
    pkl_path = ts.model_path(MODEL_DIR, _family, metric, resolution, vendor)
    if not os.path.exists(pkl_path):
        return (f"<p style='color:#c0392b'>Model not found: {os.path.basename(pkl_path)} "
                f"— run training_forecast_models.py.</p>")

    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    model     = pkg["model"]
    feat_cols = pkg["features"]
    n_lag     = pkg["lag_n"]
    cal       = pkg["calibrator"]

    dk       = _resolution_seconds(resolution)
    start_dt = _dt(start)
    end_dt   = _dt(end)
    buf_lo   = start_dt - timedelta(seconds=n_lag * dk)

    # Pull the data from the new sefer_2017 (Trip2017) table
    buf = _binned_union(metric, resolution, vendor, buf_lo, end_dt, how=_how)
    if len(buf) == 0:
        return "<p style='color:#c0392b'>No data in this range</p>"

    buf = ts.add_time_features(buf, "zaman")#-----------------
    buf = buf.sort_values("zaman").reset_index(drop=True)

    # Build the lag features on the fly
    for lag in range(1, n_lag + 1):
        buf[f"lag_{lag}"] = buf["deger"].shift(lag)

    inf = buf[buf["zaman"] >= pd.Timestamp(start_dt)].dropna(subset=feat_cols).copy()
    if len(inf) == 0:
        return "<p style='color:#c0392b'>Not enough preceding data for lag features</p>"

    X = inf[feat_cols].values.astype(float)
    # training uses xgb.train() → the model is an xgb.Booster, whose .predict()
    # needs a DMatrix (a plain numpy array raises/returns wrong). Handle both a
    # Booster and a sklearn XGBRegressor for safety.
    try:
        import xgboost as xgb
        if isinstance(model, xgb.Booster):
            inf["tahmin"] = model.predict(xgb.DMatrix(X))
        else:
            inf["tahmin"] = model.predict(X)
    except ImportError:
        inf["tahmin"] = model.predict(X)

    # Build the chart
    fig = go.Figure()
    ab = cal_mod.band(inf["tahmin"].values, cal, level)
    lower = upper = None
    if ab is not None:
        lower, upper, w = ab
        fig.add_scatter(
            x=pd.concat([inf["zaman"], inf["zaman"][::-1]]),
            y=pd.concat([pd.Series(upper), pd.Series(lower)[::-1]]),
            fill="toself", fillcolor=cal_mod.fill_color(level),
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{cal_mod.label(level)} (±{w:.2f})", showlegend=True
        )

    # Special color for XGBoost: orange/red (#D35400) — stream variant overrides
    fig.add_scatter(x=inf["zaman"], y=inf["tahmin"], mode="lines",
                    name=f"{_title} forecast", line=dict(color=_color, width=2))

    actual = buf[buf["zaman"] >= pd.Timestamp(start_dt)].copy()
    fig.add_scatter(x=actual["zaman"], y=actual["deger"], mode="lines",
                    name="actual", line=dict(color="black", width=1.5))

    mt = _metrics(inf["deger"].to_numpy(), inf["tahmin"].to_numpy())
    fig.update_layout(
        title=f"{metric} · {_title} · {resolution} · "
              f"{start_date} {start_time}→{end_date} {end_time}"
              f"<br><sub>{_metrics_label(mt)}</sub>",
        height=480, template="plotly_white", legend=dict(orientation="h", y=-0.15)
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_CFG)


def compute_svr(metric, resolution, vendor, start, end, _vf, period, k,
                 start_date, start_time, end_date, end_time, level="95"):
    """SVR point forecast + conformal confidence band."""
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_update")
    pkl_path = ts.model_path(MODEL_DIR, "svr", metric, resolution, vendor)
    if not os.path.exists(pkl_path):
        return (f"<p style='color:#c0392b'>Model not found: {os.path.basename(pkl_path)}</p>")

    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    model     = pkg["model"]
    feat_cols = pkg["features"]
    n_lag     = pkg["lag_n"]
    cal       = pkg["calibrator"]

    dk       = _resolution_seconds(resolution)
    start_dt = _dt(start)
    end_dt   = _dt(end)
    buf_lo   = start_dt - timedelta(seconds=n_lag * dk)

    buf = _binned_union(metric, resolution, vendor, buf_lo, end_dt)
    if len(buf) == 0: return "<p style='color:#c0392b'>No data</p>"

    buf = ts.add_time_features(buf, "zaman")
    buf = buf.sort_values("zaman").reset_index(drop=True)

    for lag in range(1, n_lag + 1):
        buf[f"lag_{lag}"] = buf["deger"].shift(lag)

    inf = buf[buf["zaman"] >= pd.Timestamp(start_dt)].dropna(subset=feat_cols).copy()
    if len(inf) == 0: return "<p style='color:#c0392b'>Not enough data</p>"

    # SVR needs the data to be scaled (StandardScaler). 
    # If you used a scaler during training, load it from the pkg and apply "X = scaler.transform(X)".
    X = inf[feat_cols].values.astype(float)
    if "scaler" in pkg:
        X = pkg["scaler"].transform(X)
        
    inf["tahmin"] = model.predict(X)

    fig = go.Figure()
    ab = cal_mod.band(inf["tahmin"].values, cal, level)
    if ab is not None:
        lower, upper, w = ab
        fig.add_scatter(
            x=pd.concat([inf["zaman"], inf["zaman"][::-1]]),
            y=pd.concat([pd.Series(upper), pd.Series(lower)[::-1]]),
            fill="toself", fillcolor=cal_mod.fill_color(level), line=dict(color="rgba(0,0,0,0)"), name=f"{cal_mod.label(level)}"
        )

    # Special color for SVR: purple (#8E44AD)
    fig.add_scatter(x=inf["zaman"], y=inf["tahmin"], mode="lines",
                    name="SVR forecast", line=dict(color="#8E44AD", width=2))
    
    actual = buf[buf["zaman"] >= pd.Timestamp(start_dt)].copy()
    fig.add_scatter(x=actual["zaman"], y=actual["deger"], mode="lines", name="actual", line=dict(color="black", width=1.5))

    mt = _metrics(inf["deger"].to_numpy(), inf["tahmin"].to_numpy())
    fig.update_layout(title=f"{metric} · SVR · {resolution}<br><sub>{_metrics_label(mt)}</sub>", height=480, template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs=False, config=_CFG)

# ── Fine-tuned foundation models (TimesFM 2.5 LoRA + Chronos-2 LoRA) ──────────
# Loaded via the HF Transformers / Chronos-2 interfaces the adapters were trained
# with (NOT the timesfm/chronos-t5 packages the zero-shot funcs above use).
# Requires on the server:  pip install -U transformers peft chronos-forecasting
# Adapters expected at:  finetune/timesfm_ft/  and  finetune/chronos_ft/checkpoint/
_FT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "finetune")


def _torch():
    import torch
    return torch


def _ft_device():
    t = _torch()
    return "cuda" if t.cuda.is_available() else "cpu"


def _mult32(arr, max_len=512):
    """Trim a 1-D context to the largest multiple of 32 (TimesFM 2.5 requires this)."""
    a = np.asarray(arr, dtype=np.float32)
    L = (min(len(a), max_len) // 32) * 32
    return a[-L:] if L >= 32 else a


def _load_timesfm_adapter(cache_key, adapter_dir):
    """Load base TimesFM 2.5 + a LoRA adapter dir, cache under cache_key.

    Used by BOTH the manual-FT model (finetune/timesfm_ft) and the Optuna-tuned
    model (finetune/timesfm_ft_optuna). Each gets its own cache slot, so they
    coexist in the panel. Adapter dirs hold adapter_config.json +
    adapter_model.safetensors directly (no 'checkpoint' subfolder here)."""
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    from transformers import TimesFm2_5ModelForPrediction
    from peft import PeftModel
    t = _torch(); dev = _ft_device()
    base = TimesFm2_5ModelForPrediction.from_pretrained(
        "google/timesfm-2.5-200m-transformers",
        torch_dtype=t.bfloat16 if dev == "cuda" else t.float32)
    model = PeftModel.from_pretrained(base, adapter_dir).to(dev).eval()
    _MODEL_CACHE[cache_key] = (model, dev)
    return model, dev


def _load_timesfm_ft():
    return _load_timesfm_adapter("timesfm_ft", os.path.join(_FT_DIR, "timesfm_ft"))


def _load_timesfm_ft_optuna():
    return _load_timesfm_adapter("timesfm_ft_optuna",
                                 os.path.join(_FT_DIR, "timesfm_ft_optuna"))


def _load_chronos_adapter(cache_key, adapter_dir):
    """Load base chronos-2 + a LoRA adapter dir, merge, cache under cache_key.

    Used by BOTH the manual-FT model (finetune/chronos_ft/checkpoint) and the
    Optuna-tuned model (finetune/chronos_ft_optuna/checkpoint). Each gets its own
    cache slot + its own merged pipeline, so they coexist in the panel."""
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    from chronos import BaseChronosPipeline
    from peft import PeftModel
    t = _torch(); dev = _ft_device()
    # the checkpoint is a LoRA ADAPTER (adapter_config.json + adapter_model.safetensors),
    # not a full model → load base chronos-2, attach the adapter, then merge it in.
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", device_map=dev,
        torch_dtype=t.bfloat16 if dev == "cuda" else t.float32)
    peft_model = PeftModel.from_pretrained(pipe.model, adapter_dir)
    pipe.model = peft_model.merge_and_unload()      # fold LoRA into the base weights
    _MODEL_CACHE[cache_key] = (pipe, dev)
    return pipe, dev


def _load_chronos_ft():
    return _load_chronos_adapter("chronos_ft",
                                 os.path.join(_FT_DIR, "chronos_ft", "checkpoint"))


def _load_chronos_ft_optuna():
    return _load_chronos_adapter("chronos_ft_optuna",
                                 os.path.join(_FT_DIR, "chronos_ft_optuna", "checkpoint"))


def _rolling_forecast(predict_fn, ctx_df, win_df, C=512, H=24):
    """Roll forward over the window, re-anchoring with the ACTUAL values each step."""
    history = list(ctx_df["deger"].to_numpy(dtype=float))
    actual  = win_df["deger"].to_numpy(dtype=float)
    Wn = len(actual); y_med = np.empty(Wn, dtype=float); i = 0
    while i < Wn:
        h = int(min(H, Wn - i))
        pred = np.asarray(predict_fn(np.asarray(history[-C:], dtype=float), h), dtype=float)
        y_med[i:i+h] = pred[:h]
        history.extend(actual[i:i+h].tolist())
        i += h
    return y_med


def _timesfm_ft_render(loader, title, color, metric, resolution, vendor, start, end,
                       start_date, start_time, end_date, end_time, level):
    """Shared body for the TimesFM 2.5 FT models (manual FT + Optuna FT)."""
    try:
        model, dev = loader()
    except Exception as e:
        return f"<p style='color:#c0392b'>{title} load error: {type(e).__name__}: {e}</p>"
    ctx_df, win_df = _context_window(metric, resolution, vendor, start, end, max_ctx=512)
    if ctx_df is None:
        return "<p style='color:#c0392b'>Not enough history/window data</p>"
    t = _torch()

    def _predict(ctx, h):
        x = t.tensor(_mult32(ctx)[None, :], dtype=t.float32, device=dev)
        with t.no_grad():
            return model(past_values=x).mean_predictions[0, :h].float().cpu().numpy()

    try:
        y_med = _rolling_forecast(_predict, ctx_df, win_df, C=512, H=24)
    except Exception as e:
        return f"<p style='color:#c0392b'>{title} error: {type(e).__name__}: {e}</p>"
    return _forecast_chart(win_df, y_med, None, None, title, color, level,
                           resolution, start_date, start_time, end_date, end_time)


def compute_timesfm_ft(metric, resolution, vendor, start, end, _vf, period, k,
                       start_date, start_time, end_date, end_time, level="95"):
    """TimesFM 2.5 + LoRA, fine-tuned on the taxi series (median forecast)."""
    return _timesfm_ft_render(_load_timesfm_ft, "TimesFM (FT)", "#1E8449",
                              metric, resolution, vendor, start, end,
                              start_date, start_time, end_date, end_time, level)


def compute_timesfm_ft_optuna(metric, resolution, vendor, start, end, _vf, period, k,
                              start_date, start_time, end_date, end_time, level="95"):
    """TimesFM 2.5 + LoRA, Optuna-tuned all-resolution final (finetune/timesfm_ft_optuna)."""
    return _timesfm_ft_render(_load_timesfm_ft_optuna, "TimesFM (Optuna)", "#148F77",
                              metric, resolution, vendor, start, end,
                              start_date, start_time, end_date, end_time, level)


def _chronos_ft_render(loader, title, color, metric, resolution, vendor, start, end,
                       start_date, start_time, end_date, end_time, level, how="median"):
    """Shared body for the Chronos-2 FT models (manual FT + Optuna FT)."""
    try:
        pipe, dev = loader()
    except Exception as e:
        return f"<p style='color:#c0392b'>{title} load error: {type(e).__name__}: {e}</p>"
    ctx_df, win_df = _context_window(metric, resolution, vendor, start, end, max_ctx=512, how=how)
    if ctx_df is None:
        return "<p style='color:#c0392b'>Not enough history/window data</p>"
    t = _torch()

    def _predict(ctx, h):
        x = t.tensor(np.asarray(ctx, np.float32)[None, None, :], dtype=t.float32)   # (1,1,C) CPU
        with t.no_grad():
            qq, _ = pipe.predict_quantiles(x, prediction_length=h, quantile_levels=[0.5])
        if isinstance(qq, (list, tuple)):
            qq = np.stack([a.float().cpu().numpy() if hasattr(a, "cpu") else np.asarray(a, float)
                           for a in qq], 0)
        else:
            qq = qq.float().cpu().numpy() if hasattr(qq, "cpu") else np.asarray(qq, float)
        return np.asarray(qq, dtype=float).reshape(1, -1)[0, :h]

    try:
        y_med = _rolling_forecast(_predict, ctx_df, win_df, C=512, H=24)
    except Exception as e:
        return f"<p style='color:#c0392b'>{title} error: {type(e).__name__}: {e}</p>"
    return _forecast_chart(win_df, y_med, None, None, title, color, level,
                           resolution, start_date, start_time, end_date, end_time)


def compute_chronos_ft(metric, resolution, vendor, start, end, _vf, period, k,
                       start_date, start_time, end_date, end_time, level="95"):
    """Chronos-2 + LoRA, fine-tuned on the taxi series (median forecast)."""
    return _chronos_ft_render(_load_chronos_ft, "Chronos-2 (FT)", "#7D3C98",
                              metric, resolution, vendor, start, end,
                              start_date, start_time, end_date, end_time, level)


def compute_chronos_ft_optuna(metric, resolution, vendor, start, end, _vf, period, k,
                              start_date, start_time, end_date, end_time, level="95"):
    """Chronos-2 + LoRA, Optuna-tuned all-resolution final (finetune/chronos_ft_optuna)."""
    return _chronos_ft_render(_load_chronos_ft_optuna, "Chronos-2 (Optuna)", "#B03A2E",
                              metric, resolution, vendor, start, end,
                              start_date, start_time, end_date, end_time, level)


# ── Streaming "live" variants (updated by the streaming pipeline) ────────────
# XGBoost live: reads models_update/xgboost_stream/*.pkl (written each window).
# Chronos live: uses _MODEL_CACHE['chronos_stream'], hot-swapped by the pipeline
# after each LoRA step; falls back to finetune/chronos_ft_stream/checkpoint on a
# cold start (e.g. server restarted mid-run).
def compute_xgboost_stream(metric, resolution, vendor, start, end, _vf, period, k,
                           start_date, start_time, end_date, end_time, level="95"):
    """XGBoost (Live) — the streaming pipeline's incrementally-updated model."""
    return compute_xgboost(metric, resolution, vendor, start, end, _vf, period, k,
                           start_date, start_time, end_date, end_time, level,
                           _family="xgboost_stream", _title="XGBoost (Live)", _color="#E67E22",
                           _how="mean")


def _load_chronos_stream(metric=None):
    """Live Chronos loader. Multi-metric: each metric has its own adapter dir
    finetune/chronos_ft_stream/<metric>/checkpoint and its own cache slot
    'chronos_stream::<metric>', hot-swapped by the pipeline's ChronosUpdater."""
    if metric:
        key = f"chronos_stream::{metric}"
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        return _load_chronos_adapter(
            key, os.path.join(_FT_DIR, "chronos_ft_stream", metric, "checkpoint"))
    # legacy single-dir fallback
    if "chronos_stream" in _MODEL_CACHE:
        return _MODEL_CACHE["chronos_stream"]
    return _load_chronos_adapter("chronos_stream",
                                 os.path.join(_FT_DIR, "chronos_ft_stream", "checkpoint"))


def compute_chronos_stream(metric, resolution, vendor, start, end, _vf, period, k,
                           start_date, start_time, end_date, end_time, level="95"):
    """Chronos-2 (Live) — the streaming pipeline's periodically LoRA-updated model."""
    return _chronos_ft_render(_load_chronos_stream, "Chronos-2 (Live)", "#CA6F1E",
                              metric, resolution, vendor, start, end,
                              start_date, start_time, end_date, end_time, level, how="mean")


MODELS = {
    "lgbm":       compute_lgbm,
    "xgboost":    compute_xgboost,
    "svr":        compute_svr,
    "timesfm":    compute_timesfm,
    "chronos":    compute_chronos,
    "timesfm_ft": compute_timesfm_ft,
    "timesfm_ft_optuna": compute_timesfm_ft_optuna,
    "chronos_ft": compute_chronos_ft,
    "chronos_ft_optuna": compute_chronos_ft_optuna,
    # NOTE: xgboost_stream / chronos_stream are NOT dispatched here — they render
    # on the dedicated /live/ page via streaming.live_forecast (compute_*_stream
    # remain defined for reuse but are off the panel dropdown).
}


# ── Warm-up: preload heavy models into _MODEL_CACHE at server startup ─────────
# The foundation models (Chronos / TimesFM + adapters) cold-start on their FIRST
# request — download/load hundreds of MB of weights — which makes the first
# forecast in a live demo hang for many seconds. Calling this once at startup
# (see apps.py → ready()) moves that cost off the demo's critical path.
#
# Only the loaders listed in WARMUP_MODELS are preloaded (RAM-bounded — loading
# every model at once is heavy). Default = the two Optuna models you'll demo +
# zero-shot Chronos (the slowest cold-start). Edit the list to taste.
WARMUP_MODELS = ["chronos_ft_optuna", "timesfm_ft_optuna", "chronos_zeroshot"]

_WARMUP_LOADERS = {
    "chronos_ft":        _load_chronos_ft,
    "chronos_ft_optuna": _load_chronos_ft_optuna,
    "timesfm_ft":        _load_timesfm_ft,
    "timesfm_ft_optuna": _load_timesfm_ft_optuna,
}


def _preload_chronos_zeroshot():
    """Populate _MODEL_CACHE['chronos'] the same way compute_chronos does."""
    if "chronos" in _MODEL_CACHE:
        return
    import torch
    from chronos import ChronosPipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _MODEL_CACHE["chronos"] = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-large", device_map=dev, torch_dtype=torch.float32)


def warmup_models(which=None, verbose=True):
    """Preload the models named in `which` (default WARMUP_MODELS) into the cache.

    Safe to call from a background thread at startup: every load is guarded, so a
    single failure (missing adapter, no torch, OOM) is logged and skipped without
    crashing the server. Idempotent — the loaders/cache no-op if already warm.
    """
    which = which or WARMUP_MODELS
    for name in which:
        try:
            if name == "chronos_zeroshot":
                _preload_chronos_zeroshot()
            elif name in _WARMUP_LOADERS:
                _WARMUP_LOADERS[name]()
            else:
                if verbose:
                    print(f"[warmup] unknown model '{name}' — skipped")
                continue
            if verbose:
                print(f"[warmup] loaded {name}")
        except Exception as e:
            if verbose:
                print(f"[warmup] {name} failed: {type(e).__name__}: {e}")