"""
streaming/params.py
Loads the Optuna-tuned live params (live_params.json at project root, written by
tune_live_models.py) so the cleaner + updater run with their optimized config.
Falls back to sensible defaults when the file is absent (untuned run).
"""
import os, json

_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PATH = os.path.join(_PROJECT, "live_params.json")

_DEFAULT_IF = {"contamination": 0.026, "n_estimators": 100,
               "max_samples": "auto", "max_features": 1.0}
_DEFAULT_XGB = {"eta": 0.05, "max_depth": 8, "min_child_weight": 20,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "num_boost_round": 300, "lag_n": None}   # lag_n None → derive from resolution

_cache = None


def _load():
    global _cache
    if _cache is None:
        try:
            with open(_PATH) as f:
                _cache = json.load(f)
        except Exception:
            _cache = {}
    return _cache


def if_params():
    """Isolation Forest kwargs (contamination, n_estimators, max_samples, max_features)."""
    p = _load().get("if")
    return dict(p) if p else dict(_DEFAULT_IF)


def xgb_params(metric):
    """(xgb_kwargs, num_boost_round, lag_n) for a metric — tuned if available."""
    p = (_load().get("xgb") or {}).get(metric)
    src = dict(p) if p else dict(_DEFAULT_XGB)
    src.pop("best_wape", None)
    rounds = src.pop("num_boost_round", 300)
    lag_n = src.pop("lag_n", None)
    return src, int(rounds), lag_n


def is_tuned():
    return bool(_load())
