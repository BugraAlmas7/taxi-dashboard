"""
<app>/management/commands/run_pipeline.py
Run the live streaming pipeline from the CLI.

    python manage.py run_pipeline --metric sefer --resolution 1h \
        --models xgboost,chronos --mode finetune \
        --source trip2017 --start "2017-01-01 00:00" \
        --window 1000 --step 200 --max-windows 50 --sleep 0.2

Place this file at  <your_app>/management/commands/run_pipeline.py  (e.g.
anomaly/management/commands/). It updates the "live" model slots
(xgboost_stream / chronos_stream) and writes one row per model per window to the
PipelineWindowResult table, which the dashboard polls.
"""
from django.core.management.base import BaseCommand

from ...streaming import PipelineConfig, run_pipeline


class Command(BaseCommand):
    help = "Run the streaming clean-and-update forecast pipeline."

    def add_arguments(self, p):
        # multi-metric: comma list; every window updates all of them (one live
        # model per metric) so the /live/ page's 5 panels fill from one run.
        p.add_argument("--metrics", default="sefer,total_amount",
                       help="comma list; each metric gets its own live model")
        p.add_argument("--resolution", default="1h")
        p.add_argument("--vendor", default="hepsi")
        p.add_argument("--source", default="trip2017", choices=["trip2017", "train"])
        p.add_argument("--start", default="2017-01-01 00:00")
        p.add_argument("--batch", type=int, default=200)
        p.add_argument("--window", type=int, default=1000)
        p.add_argument("--step", type=int, default=200)
        p.add_argument("--contamination", type=float, default=0.02)
        p.add_argument("--models", default="xgboost,chronos",
                       help="comma list of: xgboost,chronos")
        p.add_argument("--mode", default="finetune", choices=["finetune", "train"])
        p.add_argument("--chronos-every", type=int, default=5)
        p.add_argument("--chronos-steps", type=int, default=200)
        p.add_argument("--max-windows", type=int, default=0)
        p.add_argument("--sleep", type=float, default=0.0)
        p.add_argument("--run-name", default="stream")

    def handle(self, *a, **o):
        cfg = PipelineConfig(
            metrics=[m.strip() for m in o["metrics"].split(",") if m.strip()],
            resolution=o["resolution"], vendor=o["vendor"],
            source=o["source"], start=o["start"], batch_size=o["batch"],
            window_size=o["window"], window_step=o["step"],
            contamination=o["contamination"],
            models=[m.strip() for m in o["models"].split(",") if m.strip()],
            mode=o["mode"],
            chronos_update_every=o["chronos_every"], chronos_steps=o["chronos_steps"],
            max_windows=o["max_windows"], tick_sleep=o["sleep"],
            run_name=o["run_name"], verbose=True,
        )
        self.stdout.write(self.style.SUCCESS(
            f"▶ pipeline start · models={cfg.models} mode={cfg.mode} "
            f"metrics={cfg.metrics} res={cfg.resolution} src={cfg.source}"))
        run_pipeline(cfg)
        self.stdout.write(self.style.SUCCESS("✓ pipeline finished"))
