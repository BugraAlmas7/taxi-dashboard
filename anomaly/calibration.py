"""
calibration.py
Split-conformal band → coverage calibration (prediction interval).
Works WITHOUT re-training — the existing calibrator {width_90, width_95} is enough.

  band(prediction, cal, level)                       → symmetric fixed band + non-negative clip
  band(prediction, cal, level, local_scale=scale)    → level-proportional band (mean width kept)
"""
import numpy as np

LEVELS = {"yok": None, "90": "width_90", "95": "width_95"}
_ALPHA = {"90": 0.20, "95": 0.12}   # visual fill opacity only — not a statistic


def cqr_adjust(q_lo, q_hi, Q):
    """Widen/shrink a native quantile band by the conformal correction (CQR)."""
    q_lo = np.asarray(q_lo, dtype=float) - Q
    q_hi = np.asarray(q_hi, dtype=float) + Q
    return np.maximum(q_lo, 0.0), q_hi


def cqr_Q(y_true, q_lo, q_hi, level):
    """Compute Q from the calibration set (one-off, from an offline backtest)."""
    y_true = np.asarray(y_true, float); q_lo = np.asarray(q_lo, float); q_hi = np.asarray(q_hi, float)
    E = np.maximum(q_lo - y_true, y_true - q_hi)
    lvl = 0.90 if level == "90" else 0.95
    return float(np.quantile(E, lvl))


def valid_level(level):
    return level if level in LEVELS else "95"


def band(prediction, calibrator, level, non_negative=True, local_scale=None):
    """
    Returns (lower, upper, mean_half_width) or None (level == 'yok').
      non_negative : clip the lower bound at 0 (counts/demand can't be negative). Default on.
      local_scale  : if given, spreads the fixed width by this scale; the MEAN width
                     is preserved (no training needed). Same length as prediction.
    """
    level = valid_level(level)
    if LEVELS[level] is None:
        return None
    prediction = np.asarray(prediction, dtype=float)
    w = float(calibrator[LEVELS[level]])           # fixed half-width from training

    if local_scale is None:
        w_arr = np.full_like(prediction, w)
    else:
        scale = np.asarray(local_scale, dtype=float)
        mean_scale = float(np.mean(scale))
        # normalize scale to mean 1 → mean(w_arr) == w is preserved
        w_arr = w * (scale / (mean_scale if mean_scale else 1.0))

    lower = prediction - w_arr
    upper = prediction + w_arr
    if non_negative:
        lower = np.maximum(lower, 0.0)
    return lower, upper, float(np.mean(w_arr))


def fill_color(level):
    a = _ALPHA.get(valid_level(level), 0.12)
    return f"rgba(28,114,147,{a})"


def label(level):
    level = valid_level(level)
    return {"yok": "no calibration", "90": "90% band", "95": "95% band"}[level]