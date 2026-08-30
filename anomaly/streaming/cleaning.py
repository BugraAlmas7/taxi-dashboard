"""
streaming/cleaning.py
Isolation Forest cleaning at the RAW-TRIP level. Each window we fit a fresh IF on
the trips' numeric features and drop the flagged outliers (negative speeds,
absurd distances, impossible fares — exactly the 2015-16 pollution that wrecked
the mean-binned forecasts). Cleaning raw records BEFORE binning keeps the
downstream median series honest.

Why fit fresh per window instead of loading models_update/if/*.pkl:
  those .pkl anomaly models score a specific binned (metric,res,vendor) *series*,
  not raw trips. Here we want a per-window, unsupervised outlier filter on the
  trip feature space, so a fresh IF on the window is the right tool.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from . import params as live_params


class IsolationForestCleaner:
    def __init__(self, cfg):
        self.features = cfg.clean_features
        # tuned IF kwargs from live_params.json when present; else cfg.contamination
        self.if_kwargs = live_params.if_params()
        if not live_params.is_tuned():
            self.if_kwargs["contamination"] = cfg.contamination

    def clean(self, win_df):
        """Return (clean_df, stats). clean_df drops IF-flagged outliers and rows
        with unusable features. stats carries the counts for telemetry."""
        n_raw = len(win_df)
        cols = [c for c in self.features if c in win_df.columns]
        X = win_df[cols].apply(pd.to_numeric, errors="coerce")

        # rows with any NaN feature can't be scored → treat as unusable, drop
        usable = X.notna().all(axis=1)
        Xu = X[usable].to_numpy(dtype=float)

        keep_mask = pd.Series(False, index=win_df.index)
        n_anom = 0
        if len(Xu) >= 20:                       # IF needs a few points to be meaningful
            iso = IsolationForest(random_state=42, n_jobs=-1, **self.if_kwargs)
            pred = iso.fit_predict(Xu)          # +1 inlier, -1 outlier
            inlier = pred == 1
            keep_mask.loc[X[usable].index] = inlier
            n_anom = int((~inlier).sum())
        else:
            # too few usable rows to model → keep the usable ones as-is
            keep_mask.loc[X[usable].index] = True

        clean_df = win_df[keep_mask].copy()
        stats = {
            "n_raw": n_raw,
            "n_unusable": int((~usable).sum()),
            "n_anomaly": n_anom,
            "n_clean": len(clean_df),
            "anom_rate": round(n_anom / n_raw, 4) if n_raw else 0.0,
        }
        return clean_df, stats
