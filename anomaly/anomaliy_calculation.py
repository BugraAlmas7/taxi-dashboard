"""
anomaliy_calculation.py
One function per anomaly model + a MODELS dispatch dict.
views.py imports MODELS from here; this module holds no request logic.

Data access: the ORM only pulls raw rows; binning/summary is done entirely in
pandas (time_series.py). No SQL anywhere.
Note: anomaly binning summarizes metrics with the MEAN — the IF/SVM/MAD .pkl
files were trained that way.

Fixed data-schema strings (kept as-is, tied to trained models / plotting):
  'zaman', 'deger', 'wd', 'saat', 'dakika', 'saniye', and the internal plot
  columns 'z_skoru', 'gercek_deger', 'beklenen', 'tahmin'.
"""
import os, pickle
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .models import Trip2017, TrainingProfile, AnomalyResult
from . import time_series as ts

_CFG = {"scrollZoom": True}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dt(s):
    """'YYYY-MM-DD HH:MM' → datetime."""
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def _raw_rows(vendor, start, end, metric):
    """Pull raw rows from trip_2017 via the ORM (no SQL). Parameterized vendor filter."""
    qs = Trip2017.objects.filter(tpep_pickup_datetime__range=(_dt(start), _dt(end)))
    if vendor != "hepsi":
        qs = qs.filter(vendorid=vendor)          # vendorid is text → compare as string
    columns = ["tpep_pickup_datetime"] + ([] if metric == "sefer" else [metric])
    return qs.values(*columns)


_RES_SECONDS = {
    "1s":1,    "3s":3,    "5s":5,    "10s":10,  "15s":15,  "30s":30,
    "1m":60,   "3m":180,  "5m":300,  "10m":600, "15m":900, "30m":1800,
    "1h":3600, "3h":10800,"1d":86400,
}


def _resolution_seconds(resolution):
    return _RES_SECONDS.get(resolution, 3600)


def _binned_data(metric, resolution, vendor, start, end):
    """
    Binning consistent with training (pandas floor). Metric summary = MEAN.
    wd/saat/dakika/saniye come from the bin timestamp (Sunday=0 convention).
    """
    df = ts.bin_rows(_raw_rows(vendor, start, end, metric), metric,
                     ts.resolution_freq(resolution), how="mean")
    if len(df) == 0:
        return df
    df = ts.add_time_features(df, "zaman")
    return df[["zaman", "wd", "saat", "dakika", "saniye", "deger"]]


def _chart(df, k, metric, start_date, start_time, end_date, end_time, model_name):
    """Shared anomaly chart."""
    if "tahmin" in df.columns:
        anom = df[df["tahmin"] == -1]
    else:
        anom = df[df["z_skoru"].abs() > k]

    fig = go.Figure()
    fig.add_scatter(x=df["zaman"], y=df["gercek_deger"], mode="lines",
                    name="actual (median)", line=dict(color="black", width=1.5))
    if "beklenen" in df.columns and df["beklenen"].notna().any():
        fig.add_scatter(x=df["zaman"], y=df["beklenen"], mode="lines",
                        name="expected", line=dict(color="#1C7293", width=1.5, dash="dash"))
    fig.add_scatter(x=anom["zaman"], y=anom["gercek_deger"], mode="markers",
                    name=f"anomaly ({len(anom)})",
                    marker=dict(color="orange", size=10, symbol="x"))
    fig.update_layout(title=f"{metric} · {model_name} · k={k} · "
                            f"{start_date} {start_time}→{end_date} {end_time}",
                      height=460, template="plotly_white")
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config=_CFG)


# Reading the DB cache is OFF during development (stale data causes confusion).
CACHE_READ = False


def _cache_load(model_name_db, metric, resolution, vendor, start, end):
    if not CACHE_READ:
        return None
    existing = AnomalyResult.objects.filter(
        metric=metric, resolution=resolution, vendor=vendor, model_name=model_name_db,
        time__gte=_dt(start), time__lte=_dt(end)
    ).order_by("time")
    if existing.exists():
        rows = existing.values("time", "actual_value", "expected", "z_score")
        df = pd.DataFrame(list(rows))
        # map cache field names → internal plot columns
        return df.rename(columns={"time": "zaman", "actual_value": "gercek_deger",
                                  "expected": "beklenen", "z_score": "z_skoru"})
    return None


def _cache_save(df, model_name_db, metric, resolution, vendor):
    if not CACHE_READ:
        return
    AnomalyResult.objects.bulk_create([
        AnomalyResult(
            time=row.zaman, metric=metric, resolution=resolution, vendor=vendor,
            model_name=model_name_db, z_score=row.z_skoru,
            actual_value=row.gercek_deger,
            expected=getattr(row, "beklenen", None)
        ) for row in df.itertuples()
    ], ignore_conflicts=True)


# ── Model functions ──────────────────────────────────────────────────────────

def compute_mad(metric, resolution, vendor, start, end, _vf, period, k,
                start_date, start_time, end_date, end_time):
    """Global MAD: z-score against the 2015-16 training-profile baseline."""
    cached = _cache_load("mad", metric, resolution, vendor, start, end)
    if cached is not None:
        return _chart(cached, k, metric, start_date, start_time, end_date, end_time, "MAD")

    profile = pd.DataFrame(list(
        TrainingProfile.objects
        .filter(metric=metric, resolution=resolution, vendor=vendor)
        .values("wd", "tb", "expected", "mad")
    ))
    if len(profile) == 0:
        return "<p style='color:#c0392b'>No training profile for this metric/resolution</p>"

    # floor at the (coarse) period; metric summary = MEAN
    df = ts.bin_rows(_raw_rows(vendor, start, end, metric), metric,
                     ts.period_freq(period), how="mean")
    if len(df) == 0:
        return "<p style='color:#c0392b'>No data in this range</p>"

    df["wd"] = (pd.to_datetime(df["zaman"]).dt.dayofweek + 1) % 7    # Sunday=0
    df["tb"] = ts.seconds_of_day(df["zaman"])

    profile["tb"] = profile["tb"].map(ts.tb_to_seconds).astype(int)
    profile["wd"] = profile["wd"].astype(int)
    df = df.merge(profile, on=["wd", "tb"], how="left")
    df["z_skoru"]      = 0.6745 * (df["deger"] - df["expected"]) / df["mad"].replace(0, 1e-9)
    df["gercek_deger"] = df["deger"]
    df["beklenen"]     = df["expected"]

    _cache_save(df, "mad", metric, resolution, vendor)
    return _chart(df, k, metric, start_date, start_time, end_date, end_time, "MAD")


def compute_mad_local(metric, resolution, vendor, start, end, _vf, period, k,
                      start_date, start_time, end_date, end_time):
    """Local MAD: rolling window with ±1 hour buffer, point-spike detection."""
    fmt      = "%Y-%m-%d %H:%M"
    start_dt = datetime.strptime(start, fmt)
    end_dt   = datetime.strptime(end, fmt)
    buf_lo   = (start_dt - timedelta(hours=1)).strftime(fmt)
    buf_hi   = (end_dt + timedelta(hours=1)).strftime(fmt)

    buf = ts.bin_rows(_raw_rows(vendor, buf_lo, buf_hi, metric), metric,
                      ts.period_freq(period), how="mean")
    if len(buf) == 0:
        return "<p style='color:#c0392b'>No data in this range</p>"

    buf["zaman"] = pd.to_datetime(buf["zaman"])
    w = max(20, len(buf) // 10)
    buf["rolling_med"] = buf["deger"].rolling(w, center=True, min_periods=1).median()
    buf["rolling_mad"] = (buf["deger"] - buf["rolling_med"]).abs() \
                          .rolling(w, center=True, min_periods=1).median()
    buf["z_skoru"] = 0.6745 * (buf["deger"] - buf["rolling_med"]) / \
                     buf["rolling_mad"].replace(0, 1e-9)

    df = buf[(buf["zaman"] >= pd.Timestamp(start)) &
             (buf["zaman"] <= pd.Timestamp(end))].copy()
    if len(df) == 0:
        return "<p style='color:#c0392b'>No data in this range</p>"

    df["gercek_deger"] = df["deger"]
    df["beklenen"]     = df["rolling_med"]

    anom = df[df["z_skoru"].abs() > k]
    fig = go.Figure()
    fig.add_scatter(x=df["zaman"], y=df["gercek_deger"], mode="lines",
                    name="actual", line=dict(color="black", width=1.5))
    fig.add_scatter(x=df["zaman"], y=df["beklenen"], mode="lines",
                    name=f"rolling median (w={w})",
                    line=dict(color="#1C7293", width=1.5, dash="dash"))
    fig.add_scatter(x=anom["zaman"], y=anom["gercek_deger"], mode="markers",
                    name=f"spike ({len(anom)})",
                    marker=dict(color="orange", size=10, symbol="x"))
    fig.update_layout(
        title=f"{metric} · Local MAD (window={w}) · k={k} · "
              f"{start_date} {start_time}→{end_date} {end_time}",
        height=460, template="plotly_white"
    )
    return fig.to_html(full_html=False, include_plotlyjs="cdn", config=_CFG)


def _robust_z(score):
    med = np.median(score)
    mad = np.median(np.abs(score - med))
    return 0.6745 * (score - med) / (mad if mad else 1e-9)


def _local_dev_scale(df, w=11):
    local_med = df["deger"].rolling(w, center=True, min_periods=1).median()
    dev = df["deger"] - local_med
    scale = max(float(dev.abs().median()), 1e-9)
    return dev, scale


def _anomaly_mask(df, k, w_model=0.4, w_value=0.6):
    if len(df) < 5:
        return (df["z_skoru"] > k)
    dev, scale = _local_dev_scale(df)
    value_z = (0.6745 * dev / scale).clip(lower=0)
    model_z = df["z_skoru"].clip(lower=0)
    score = w_model * model_z + w_value * value_z
    df["z_skoru"] = score
    return score > k

def _load_anom_model(model_dir, prefix, metric, resolution, vendor):
    """
    Load an anomaly .pkl and return ((model, scaler, feats), None) or
    (None, error_html).

    ts.model_path() now returns a PATH STRING (it does NOT open the file), so
    the loading + unpacking happens here. The pickle may be either a tuple
    (model, scaler, feats) or a dict {model, scaler, features} — both work.
    """
    pkl_path = ts.model_path(model_dir, prefix, metric, resolution, vendor)
    if not os.path.exists(pkl_path):
        return None, (f"<p style='color:#c0392b'>Model not found: "
                      f"{os.path.basename(pkl_path)}</p>")
    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)

    if isinstance(pkg, dict):
        model  = pkg.get("model")
        scaler = pkg.get("scaler")
        feats  = pkg.get("features") or pkg.get("feats") or pkg.get("feature_cols")
    else:                                   # tuple / list → (model, scaler, feats)
        model, scaler, feats = pkg[0], pkg[1], pkg[2]

    if model is None or scaler is None or not feats:
        return None, ("<p style='color:#c0392b'>Model file is malformed "
                      "(missing model / scaler / features)</p>")
    return (model, scaler, list(feats)), None


# ── Isolation Forest ─────────────────────────────────────────────────────────

def compute_if(metric, resolution, vendor, start, end, _vf, period, k,
               start_date, start_time, end_date, end_time):
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_update")
    loaded, err = _load_anom_model(MODEL_DIR, "if", metric, resolution, vendor)
    if err:
        return err
    model, scaler, feats = loaded

    cached = _cache_load("if", metric, resolution, vendor, start, end)
    if cached is not None:
        cached["tahmin"] = np.where(cached["z_skoru"] > k, -1, 1)
        return _chart(cached, k, metric, start_date, start_time, end_date, end_time, "Isolation Forest")

    df = _binned_data(metric, resolution, vendor, start, end)
    if len(df) == 0:
        return "<p style='color:#c0392b'>No data in this range</p>"
    df = df.dropna(subset=feats).reset_index(drop=True)
    if len(df) == 0:
        return "<p style='color:#c0392b'>No valid (non-empty) values in this range</p>"

    X = scaler.transform(df[feats].values)
    raw = -model.decision_function(X)
    df["z_skoru"] = _robust_z(raw)
    df["gercek_deger"] = df["deger"]
    df["beklenen"]     = np.nan
    df["tahmin"] = np.where(_anomaly_mask(df, k), -1, 1)

    _cache_save(df, "if", metric, resolution, vendor)
    return _chart(df, k, metric, start_date, start_time, end_date, end_time, "Isolation Forest")


# ── One-Class SVM ────────────────────────────────────────────────────────────

def compute_svm(metric, resolution, vendor, start, end, _vf, period, k,
                start_date, start_time, end_date, end_time):
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_update")
    loaded, err = _load_anom_model(MODEL_DIR, "ocsvm", metric, resolution, vendor)
    if err:
        return err
    model, scaler, feats = loaded

    cached = _cache_load("svm", metric, resolution, vendor, start, end)
    if cached is not None:
        cached["tahmin"] = np.where(cached["z_skoru"] > k, -1, 1)
        return _chart(cached, k, metric, start_date, start_time, end_date, end_time, "One-Class SVM")

    df = _binned_data(metric, resolution, vendor, start, end)
    if len(df) == 0:
        return "<p style='color:#c0392b'>No data in this range</p>"
    df = df.dropna(subset=feats).reset_index(drop=True)
    if len(df) == 0:
        return "<p style='color:#c0392b'>No valid (non-empty) values in this range</p>"

    X = scaler.transform(df[feats].values)
    # neutralize time axes → score comes from 'deger' only (kills edge artifacts)
    for i, f in enumerate(feats):
        if f in ("wd", "saat", "dakika", "saniye"):
            X[:, i] = 0.0

    raw = -model.decision_function(X)
    df["z_skoru"] = _robust_z(raw)
    df["gercek_deger"] = df["deger"]
    df["beklenen"]     = np.nan
    df["tahmin"] = np.where(_anomaly_mask(df, k), -1, 1)

    _cache_save(df, "svm", metric, resolution, vendor)
    return _chart(df, k, metric, start_date, start_time, end_date, end_time, "One-Class SVM")


def compute_dbscan(metric, resolution, vendor, start, end, _vf, period, k,
                   start_date, start_time, end_date, end_time):
    cached = _cache_load("dbscan", metric, resolution, vendor, start, end)
    if cached is not None:
        cached["tahmin"] = cached["z_skoru"].apply(lambda x: -1 if x > 0 else 1)
        return _chart(cached, k, metric, start_date, start_time, end_date, end_time, "DBSCAN")

    df = _binned_data(metric, resolution, vendor, start, end)
    if len(df) == 0:
        return "<p style='color:#c0392b'>No data in this range</p>"

    feats = ["deger", "wd", "saat", "dakika", "saniye"]
    df = df.dropna(subset=feats).reset_index(drop=True)
    if len(df) == 0:
        return "<p style='color:#c0392b'>No valid (non-empty) values in this range</p>"

    X = StandardScaler().fit_transform(df[feats].values)
    k_nn = min(5, len(X) - 1)
    dists, _ = NearestNeighbors(n_neighbors=k_nn).fit(X).kneighbors(X)
    eps = max(float(np.percentile(dists[:, -1], 90)), 0.3)
    min_s = max(3, len(X) // 20)
    labels = DBSCAN(eps=eps, min_samples=min_s).fit_predict(X)

    df["gercek_deger"] = df["deger"]
    df["beklenen"]     = np.nan
    df["z_skoru"]      = np.where(labels == -1, 3.0, 0.0)   # noise → high model_z
    df["tahmin"]       = np.where(_anomaly_mask(df, k), -1, 1)

    _cache_save(df, "dbscan", metric, resolution, vendor)
    return _chart(df, k, metric, start_date, start_time, end_date, end_time,
                  f"DBSCAN (eps={eps:.2f}, min_s={min_s})")

# ── Dispatch table ───────────────────────────────────────────────────────────

MODELS = {
    "mad":       compute_mad,
    "mad_lokal": compute_mad_local,
    "if":        compute_if,
    "svm":       compute_svm,
    "dbscan":    compute_dbscan,
}