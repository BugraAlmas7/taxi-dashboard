"""
training_forecast_models.py
- All resolutions added.
- XGBoost prefix issue fixed.
- Adapted to the models_update layout.
"""
import os, sys, pickle, gc, argparse

# ── Bootstrap Django ORM ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
import django
django.setup()

import numpy as np
import pandas as pd
import lightgbm as lgb
from django.db.models import Count, F, Func, Aggregate, FloatField, DateTimeField
from anomaly.models import TrainTrip


# ── MEDIAN binning, DB-side ──────────────────────────────────────────────────
# Metric summary is now the MEDIAN (via Postgres PERCENTILE_CONT), robust to the
# outliers polluting the 2015-16 training table (negative / thousands-of-mph
# speeds). MUST match inference: forecast_calculation._binned_union uses pandas
# median. percentile_cont skips NULLs automatically.
class Median(Aggregate):
    function = "PERCENTILE_CONT"
    name = "median"
    output_field = FloatField()
    template = "%(function)s(0.5) WITHIN GROUP (ORDER BY %(expressions)s)"
    allow_distinct = False


# Bucket every row into a [dk]-second bin anchored at the Unix epoch (1970-01-01),
# the SAME anchor pandas .floor() uses → identical bin edges to inference, for
# EVERY resolution (1m … 1d). This also fixes the old bug where 3m/5m/3h were
# trained at minute/hour granularity. Requires PostgreSQL 14+ (date_bin).
def _bin_expr(dk):
    return Func(
        F("tpep_pickup_datetime"),
        function="date_bin",
        template=("date_bin(INTERVAL '%(dk)s seconds', %(expressions)s, "
                  "TIMESTAMP '1970-01-01 00:00:00')"),
        dk=dk,
        output_field=DateTimeField(),
    )

try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception as e:
    _HAS_XGB = False
    print(f"[warn] xgboost not available ({e})")

try:
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    _HAS_SVR = True
except Exception as e:
    _HAS_SVR = False
    print(f"[warn] scikit-learn not available ({e})")

_ap = argparse.ArgumentParser()
_ap.add_argument("--vendor",  default=None)
_ap.add_argument("--models",  default="lgbm,xgb,svr")
_ap.add_argument("--threads", type=int, default=5)
_ap.add_argument("--svr-cap", type=int, default=15000)
_args = _ap.parse_args()

WANT = {m.strip().lower() for m in _args.models.split(",") if m.strip()}
DO_LGBM = "lgbm" in WANT
DO_XGB  = "xgb"  in WANT and _HAS_XGB
DO_SVR  = "svr"  in WANT and _HAS_SVR

# Model output folder layout
MODEL_DIR = os.path.join(PROJECT_DIR, "models_update")
print(f"Model dir: {MODEL_DIR}")

METRICS = [
    "sefer", "passenger_count", "trip_distance", "fare_amount",
    "tip_amount", "tolls_amount", "total_amount", 
    "trip_duration_minutes", "trip_speed_mph", "price_per_distance",
    "hourly_trip_volume", "hourly_avg_speed"
]
AVG_METRICS  = METRICS[1:]
VENDOR_LIST  = [_args.vendor] if _args.vendor else ["hepsi", "1", "2"]

N_THREADS = max(1, _args.threads)
SVR_CAP   = max(2000, _args.svr_cap)
SVR_VAL_CAP = 5000

# All previously-missing resolutions added
RES_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "10m": 600, "15m": 900, "30m": 1800,
    "1h": 3600, "3h": 10800, "1d": 86400
}
# TRUNC dict removed — binning is now done by _bin_expr(dk) (date_bin) for every
# resolution, so 3m/5m/3h are binned at the correct width instead of minute/hour.

USE_TIME_FEATS = True
RNG = np.random.default_rng(42)

def _params(dk):
    if dk < 3600:      return 800_000,   160_000, 60, 400
    elif dk < 86400:   return 1_500_000, 300_000, 48, 500
    else:              return 1_500_000, 300_000, 14, 500

def time_feats(dk):
    if not USE_TIME_FEATS: return []
    if dk >= 86400: return ["wd"]
    if dk >= 3600:  return ["wd", "saat"]
    return ["wd", "saat", "dakika"]

def lag_matrix(y, idx, n_lag):
    X = np.empty((len(idx), n_lag), dtype=np.float32)
    for j in range(n_lag):
        X[:, j] = y[idx - (j + 1)]
    return X

def binned_frame(vendor, resolution):
    qs = TrainTrip.objects.all()
    if vendor != "hepsi":
        qs = qs.filter(vendorid=vendor)
    dk = RES_SECONDS[resolution]

    # sefer = row count per bin; every other metric = MEDIAN over the bin.
    agg = {"sefer": Count("id")}
    for m in AVG_METRICS:
        agg[f"m_{m}"] = Median(m)

    rows = (qs.annotate(_bin=_bin_expr(dk))
              .values("_bin")
              .annotate(**agg)
              .order_by("_bin"))
    df = pd.DataFrame(list(rows))
    if len(df) == 0: return df
    
    df = df.rename(columns={f"m_{m}": m for m in AVG_METRICS})
    df["_bin"] = pd.to_datetime(df["_bin"])
    df = df.sort_values("_bin").reset_index(drop=True)

    if USE_TIME_FEATS:
        idx = df["_bin"]
        df["wd"] = (idx.dt.dayofweek + 1) % 7
        df["saat"] = idx.dt.hour
        df["dakika"] = idx.dt.minute
    return df

def _fit_calibrator(resid):
    return {"width_90": float(np.quantile(resid, 0.90)),
            "width_95": float(np.quantile(resid, 0.95))}

def _save(prefix, obj, metric, resolution, vendor):
    # The prefix (if, svm, xgboost, ...) is used directly as the subfolder name
    fname = f"{prefix}_{metric}_{resolution}_{vendor}.pkl"
    out = os.path.join(MODEL_DIR, prefix, fname)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(obj, f)

for vendor in VENDOR_LIST:
    print(f"{'='*55}\nVENDOR: {vendor}")

    for resolution, dk in RES_SECONDS.items():
        N_TRAIN, N_VAL, n_lag, N_BOOST = _params(dk)
        z_feats = time_feats(dk)

        print(f"  [{vendor}] {resolution} aggregating (DB-side)...", end=" ", flush=True)
        binned = binned_frame(vendor, resolution)
        if len(binned) == 0:
            print("no data.")
            continue
        print(f"bins={len(binned):,}  N={N_TRAIN//1000}K lag={n_lag} boost={N_BOOST}", end="  ", flush=True)

        n_lgbm = n_xgb = n_svr = 0
        for metric in METRICS:
            series = binned[metric]
            mask = series.notna().to_numpy()
            L = int(mask.sum())
            if L < n_lag + 500: continue

            y  = series.to_numpy(dtype=np.float32)[mask]
            zf = {f: binned[f].to_numpy(dtype=np.int16)[mask] for f in z_feats}

            split = int(L * 0.8)
            tr_pool = np.arange(n_lag, split)
            val_pool = np.arange(split, L)
            if len(tr_pool) < 100 or len(val_pool) < 50: continue

            tr_idx = RNG.choice(tr_pool, min(N_TRAIN, len(tr_pool)), replace=False) if len(tr_pool) > N_TRAIN else tr_pool.copy()
            val_idx = RNG.choice(val_pool, min(N_VAL, len(val_pool)), replace=False) if len(val_pool) > N_VAL else val_pool.copy()
            tr_idx.sort(); val_idx.sort()

            X_tr = lag_matrix(y, tr_idx, n_lag)
            X_val = lag_matrix(y, val_idx, n_lag)
            if z_feats:
                X_tr = np.hstack([X_tr] + [zf[f][tr_idx].reshape(-1,1).astype(np.float32) for f in z_feats])
                X_val = np.hstack([X_val] + [zf[f][val_idx].reshape(-1,1).astype(np.float32) for f in z_feats])
            y_tr, y_val = y[tr_idx], y[val_idx]
            feat_cols = [f"lag_{i}" for i in range(1, n_lag + 1)] + z_feats

            base = {"features": feat_cols, "lag_n": n_lag, "coz": resolution, "metrik": metric, "vendor": vendor}

            if DO_LGBM:
                dtrain = lgb.Dataset(X_tr, label=y_tr)
                dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
                params = {"objective":"regression", "metric":"rmse", "learning_rate":0.05, "num_leaves":63, "min_child_samples":20, "feature_fraction":0.8, "bagging_fraction":0.8, "bagging_freq":5, "max_bin":127, "num_threads":N_THREADS, "force_col_wise":True, "verbosity":-1}
                m = lgb.train(params, dtrain, num_boost_round=N_BOOST, valid_sets=[dval], callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)])
                resid = np.abs(y_val - m.predict(X_val))
                _save("lgbm", {"kind":"lgbm", "model":m, "calibrator":_fit_calibrator(resid), **base}, metric, resolution, vendor)
                n_lgbm += 1

            if DO_XGB:
                dtr = xgb.DMatrix(X_tr, label=y_tr)
                dva = xgb.DMatrix(X_val, label=y_val)
                xparams = {"objective":"reg:squarederror", "eval_metric":"rmse", "eta":0.05, "max_depth":8, "min_child_weight":20, "subsample":0.8, "colsample_bytree":0.8, "max_bin":127, "tree_method":"hist", "nthread":N_THREADS}
                booster = xgb.train(xparams, dtr, num_boost_round=N_BOOST, evals=[(dva, "val")], early_stopping_rounds=40, verbose_eval=False)
                resid = np.abs(y_val - booster.predict(dva))
                
                # name fixed to XGBOOST
                _save("xgboost", {"kind":"xgb", "model":booster, "calibrator":_fit_calibrator(resid), **base}, metric, resolution, vendor)
                n_xgb += 1

            if DO_SVR:
                s_tr = RNG.choice(len(tr_idx), min(SVR_CAP, len(tr_idx)), replace=False) if len(tr_idx) > SVR_CAP else np.arange(len(tr_idx))
                s_va = RNG.choice(len(val_idx), min(SVR_VAL_CAP, len(val_idx)), replace=False) if len(val_idx) > SVR_VAL_CAP else np.arange(len(val_idx))
                Xs_tr, ys_tr = X_tr[s_tr], y_tr[s_tr]
                Xs_va, ys_va = X_val[s_va], y_val[s_va]

                scaler = StandardScaler().fit(Xs_tr)
                svr = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1, cache_size=500)
                svr.fit(scaler.transform(Xs_tr), ys_tr)
                resid = np.abs(ys_va - svr.predict(scaler.transform(Xs_va)))
                _save("svr", {"kind":"svr", "model":svr, "scaler":scaler, "calibrator":_fit_calibrator(resid), **base}, metric, resolution, vendor)
                n_svr += 1

            del y, zf, X_tr, X_val, y_tr, y_val
            gc.collect()

        del binned
        gc.collect()
        print(f"saved → LGBM({n_lgbm}) XGB({n_xgb}) SVR({n_svr})")

print(f"\n{'='*55}\nDONE! Updated models under: {MODEL_DIR}")