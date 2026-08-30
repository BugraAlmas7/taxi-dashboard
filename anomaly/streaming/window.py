"""
streaming/window.py
Sliding window over the raw-trip stream. Keeps the last `window_size` records in
a ring buffer and emits a window snapshot every `window_step` new records — so
consecutive windows overlap by (W - S) records, which is what lets the models see
each region more than once as it slides.
"""
from collections import deque

import pandas as pd


class SlidingWindow:
    def __init__(self, cfg):
        self.W = cfg.window_size
        self.S = cfg.window_step
        self._buf = deque(maxlen=self.W)
        self._since_emit = 0
        self._seen = 0

    def push(self, rows):
        """Add a batch; return a list of window DataFrames ready to process
        (usually 0 or 1, but a big batch can trigger several)."""
        emitted = []
        for r in rows:
            self._buf.append(r)
            self._since_emit += 1
            self._seen += 1
            # emit once the window is full AND S new records have arrived
            if len(self._buf) >= self.W and self._since_emit >= self.S:
                emitted.append(self._snapshot())
                self._since_emit = 0
        return emitted

    def flush(self):
        """Emit whatever is left (partial last window) at end of stream."""
        if len(self._buf):
            return [self._snapshot()]
        return []

    def _snapshot(self):
        df = pd.DataFrame(list(self._buf))
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        return df.sort_values("tpep_pickup_datetime").reset_index(drop=True)

    @property
    def total_seen(self):
        return self._seen


class TimeBlocker:
    """Accumulates raw trips until the simulated clock advances by `block_hours`,
    then emits the whole block (e.g. a full day) as one DataFrame. This is the
    nightly-batch design: clean + retrain once per day, not per record window."""
    def __init__(self, cfg):
        self.block_secs = cfg.block_hours * 3600
        self._rows = []
        self._start = None

    def push(self, rows):
        out = []
        for r in rows:
            t = r["tpep_pickup_datetime"]
            if self._start is None:
                self._start = t
            if (t - self._start).total_seconds() >= self.block_secs:
                out.append(self._emit())
                self._start = t
            self._rows.append(r)
        return out

    def flush(self):
        return [self._emit()] if self._rows else []

    def _emit(self):
        df = pd.DataFrame(self._rows)
        df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
        df = df.sort_values("tpep_pickup_datetime").reset_index(drop=True)
        self._rows = []
        return df
