"""
streaming/orchestrator.py
The loop that wires everything together:

  simulator → sliding window → IF clean → forecast updaters → persist result row

Each processed window produces one PipelineWindowResult per active model, so the
dashboard can plot WAPE-over-time and anomaly counts live.
"""
import time

from .config import PipelineConfig
from .simulator import StreamSimulator
from .window import TimeBlocker
from .cleaning import IsolationForestCleaner
from .updaters import UPDATERS


class Orchestrator:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.sim = StreamSimulator(cfg)
        self.win = TimeBlocker(cfg)          # daily block (block_hours) → clean + retrain
        self.cleaner = IsolationForestCleaner(cfg)
        # one updater per (metric × model): every window updates all of them
        self.updaters = [UPDATERS[m](cfg, metric)
                         for metric in cfg.metrics
                         for m in cfg.models if m in UPDATERS]
        self.window_index = 0

    def run(self, on_result=None, should_stop=None):
        """Consume the stream to exhaustion (or cfg.max_windows). `on_result(dict)`
        is called once per processed window. `should_stop()` (optional) is checked
        each window — return True to stop the pipeline cooperatively (Stop button)."""
        try:
            for batch in self.sim:
                if should_stop and should_stop():
                    return
                for win_df in self.win.push(batch):
                    self._process(win_df, on_result)
                    if should_stop and should_stop():
                        return
                    if self.cfg.max_windows and self.window_index >= self.cfg.max_windows:
                        return
        except StopIteration:
            pass
        for win_df in self.win.flush():          # partial last window
            self._process(win_df, on_result)

    def _process(self, win_df, on_result):
        clean_df, stats = self.cleaner.clean(win_df)
        ts_start = win_df["tpep_pickup_datetime"].iloc[0]
        ts_end = win_df["tpep_pickup_datetime"].iloc[-1]

        model_results = []
        for up in self.updaters:
            try:
                model_results.append(up.update(clean_df, self.window_index))
            except Exception as e:
                model_results.append({"model": getattr(up, "label", "?"),
                                      "status": "error", "error": f"{type(e).__name__}: {e}"})

        payload = {
            "run_name": self.cfg.run_name,
            "window_index": self.window_index,
            "ts_start": ts_start, "ts_end": ts_end,
            "resolution": self.cfg.resolution, "vendor": self.cfg.vendor,
            **stats,
            "models": model_results,   # each carries its own "metric"
        }
        self._persist(payload)
        if self.cfg.verbose:
            self._log(payload)
        if on_result:
            on_result(payload)

        self.window_index += 1
        if self.cfg.tick_sleep:
            time.sleep(self.cfg.tick_sleep)

    def _persist(self, payload):
        """Write one row per model to the live-results table. Imported lazily so
        the module is usable in tests without Django set up."""
        try:
            from ..models import PipelineWindowResult
        except Exception:
            return
        rows = []
        for mr in payload["models"]:
            rows.append(PipelineWindowResult(
                run_name=payload["run_name"], window_index=payload["window_index"],
                ts_start=payload["ts_start"], ts_end=payload["ts_end"],
                metric=mr.get("metric", "?"), resolution=payload["resolution"],
                vendor=payload["vendor"],
                n_raw=payload["n_raw"], n_anomaly=payload["n_anomaly"],
                n_clean=payload["n_clean"], anom_rate=payload["anom_rate"],
                model_name=mr.get("model", "?"), mode=mr.get("mode", ""),
                status=mr.get("status", ""),
                wape=mr.get("wape"), mae=mr.get("mae"),
                n_train=mr.get("n_train"), detail=str(mr.get("error", ""))[:300],
            ))
        if rows:
            PipelineWindowResult.objects.bulk_create(rows)

    def _log(self, p):
        head = (f"[day {p['window_index']:>3}] {p['ts_start']:%Y-%m-%d}"
                f"  raw={p['n_raw']} anom={p['n_anomaly']}({p['anom_rate']:.1%}) "
                f"clean={p['n_clean']}")
        parts = []
        for mr in p["models"]:
            w = mr.get("wape")
            tag = f"{mr.get('metric','?')}:{mr['model'].replace('_stream','')}"
            if mr.get("status") == "ok" and w is not None:
                parts.append(f"{tag}={w:.2f}")
            else:
                parts.append(f"{tag}·{mr.get('status')}")
        print(head + "  |  " + "  ".join(parts))


def run_pipeline(cfg: PipelineConfig, on_result=None):
    Orchestrator(cfg).run(on_result=on_result)
