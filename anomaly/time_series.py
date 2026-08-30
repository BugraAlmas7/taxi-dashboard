"""
time_series.py
──────────────
Time-series binning + summary — done ENTIRELY in pandas. No SQL here:
the ORM only pulls raw rows (.values(...)); binning / median / mean / count
happen on the pandas side.

Why pandas floor produces the SAME bin edges as the previous binning:
  The old binning anchor ('2000-01-03', midnight) and pandas floor's epoch
  anchor (1970-01-01, midnight) differ by a whole number of days. Every bin
  width used in the panel (1..30 s, 1..30 min, 1/3 h, 1 day) divides a day
  exactly, so both anchors yield identical edges → consistency with the
  trained models is preserved.

NOTE on fixed data-schema strings:
  The binned frame columns 'zaman'/'deger' and the time-feature columns
  'wd'/'saat'/'dakika'/'saniye' are kept as-is on purpose: they are part of
  the trained-model feature vocabulary (IF/SVM/LightGBM .pkl files) and the
  database schema. Renaming them would require re-training the models.
"""
import os
import glob
import pandas as pd


# resolution code → pandas frequency (for floor)
_RES_FREQ = {
    "1s":"1s",   "3s":"3s",   "5s":"5s",   "10s":"10s", "15s":"15s", "30s":"30s",
    "1m":"1min", "3m":"3min", "5m":"5min", "10m":"10min","15m":"15min","30m":"30min",
    "1h":"1h",   "3h":"3h",   "1d":"1D",
}

# coarse period (second/minute/hour/day) → pandas frequency
_PERIOD_FREQ = {"second":"1s", "minute":"1min", "hour":"1h", "day":"1D"}

# On-disk layout is a single `models_update/` root with ONE folder per model,
# named by its short key:  models_update/{if, lgbm, svm, svr, xgboost}/*.pkl
# `prefix` is the caller's model key; map it to the actual folder name here.
# (Only the one-class SVM differs: callers pass "ocsvm" but the folder is "svm".)
_MODEL_FOLDER = {
    "lgbm":    "lgbm",
    "xgboost": "xgboost",
    "svr":     "svr",
    "if":      "if",
    "svm":     "svm",
    "ocsvm":   "svm",
}


def resolution_freq(resolution):
    return _RES_FREQ.get(resolution, "1h")


def period_freq(period):
    return _PERIOD_FREQ.get(period, "1h")


def model_path(model_dir, prefix, metric, resolution, vendor):
    """
    Resolve the .pkl for (prefix, metric, resolution, vendor).

    Layout:  <model_dir>/<family>/<name>.pkl   where <family> is the on-disk
    folder name (see _MODEL_FOLDER) and <name> is "{p}_{metric}_{res}_{vendor}".

    Robust to how the training script spelled the file prefix: it tries the
    caller's prefix AND the folder name, then, as a last resort, globs the
    folder for any file ending in "{metric}_{resolution}_{vendor}.pkl".
    Returns the first existing path, or the primary candidate (so a
    "model not found" message still shows a sensible filename).
    """
    family = _MODEL_FOLDER.get(prefix, prefix)
    folder = os.path.join(model_dir, family)

    name_prefixes = []
    for p in (prefix, family):                 # e.g. "ocsvm" then "svm"
        if p not in name_prefixes:
            name_prefixes.append(p)

    candidates = [os.path.join(folder, f"{p}_{metric}_{resolution}_{vendor}.pkl")
                  for p in name_prefixes]
    for c in candidates:
        if os.path.exists(c):
            return c

    # last resort: any file in the family folder with the right suffix
    hits = glob.glob(os.path.join(folder, f"*{metric}_{resolution}_{vendor}.pkl"))
    if hits:
        return sorted(hits)[0]

    return candidates[0]


def bin_rows(rows, metric, freq, how="median"):
    """
    Bins raw rows (list of dicts) in pandas.
    rows   : QuerySet.values(...) result OR an already-built DataFrame
             (the forecast side merges two tables and passes a DataFrame)
    metric : "sefer" → row COUNT per bin; otherwise → summary of that column
    freq   : pandas frequency (from resolution_freq / period_freq)
    how    : "median" | "mean"  (for metric != "sefer")
    Returns: DataFrame[zaman, deger]  (non-empty bins only)
    """
    # If rows is already a DataFrame use it (list(DataFrame) yields column names!);
    # otherwise build a DataFrame from the QuerySet/dict list.
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    if len(df) == 0:
        return pd.DataFrame(columns=["zaman", "deger"])

    ts = pd.to_datetime(df["tpep_pickup_datetime"])
    df["zaman"] = ts.dt.floor(freq)

    if metric == "sefer":
        out = df.groupby("zaman").size().reset_index(name="deger")
    else:
        # Force numeric even if the column arrives as text (robust to DB type)
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        series = df.groupby("zaman")[metric]
        out = (series.mean() if how == "mean" else series.median()).reset_index(name="deger")

    out["deger"] = out["deger"].astype(float)
    return out.sort_values("zaman").reset_index(drop=True)


def add_time_features(df, column="zaman"):
    """
    Derives wd/saat/dakika/saniye from the bin timestamp.

    DOW convention: the training profile / .pkl models were built with
    Sunday=0..Saturday=6. pandas dayofweek is Monday=0..Sunday=6, so we convert:
        pg_dow = (dayofweek + 1) % 7
    """
    z = pd.to_datetime(df[column])
    df["wd"]     = (z.dt.dayofweek + 1) % 7      # Sunday = 0
    df["saat"]   = z.dt.hour.astype(int)
    df["dakika"] = z.dt.minute.astype(int)
    df["saniye"] = z.dt.second.astype(int)       # 0 for minute+ bins; consumers expect the column
    return df


def seconds_of_day(timestamps):
    """Bin timestamp → seconds since midnight (same scale as TrainingProfile.tb)."""
    z = pd.to_datetime(timestamps)
    return (z.dt.hour * 3600 + z.dt.minute * 60 + z.dt.second).astype(int)


def tb_to_seconds(v):
    """TrainingProfile.tb (TimeField or int) → seconds since midnight."""
    if hasattr(v, "hour"):                 # datetime.time
        return v.hour * 3600 + v.minute * 60 + v.second
    return int(v)