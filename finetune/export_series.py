"""
export_series.py
────────────────
Export the MEDIAN-binned time series (same binning as inference / training) to a
single long-format parquet, ready for local/Colab fine-tuning.

Holdout = the LAST --test-days days of the 2015-16 table (TrainTrip / sefer_egitim).
The 2017 table is IGNORED. Bins before the cutoff → train, on/after → val.

Binning is identical to training_forecast_models.py: Postgres date_bin (epoch
anchor) + PERCENTILE_CONT(0.5). This keeps train/serve/fine-tune consistent.

Output columns (long format, one row per bin):
    series_id, metric, resolution, vendor, split, ts, value

Run:
    python export_series.py --out taxi_series.parquet
    python export_series.py --out taxi_series.parquet --test-days 120
    python export_series.py --out taxi_series.parquet --resolutions 1h --vendors hepsi
"""
import os, sys, argparse

# ── Bootstrap Django ORM ─────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
import django
django.setup()

import pandas as pd
from datetime import timedelta
from django.db.models import Count, F, Func, Aggregate, FloatField, DateTimeField
from anomaly.models import TrainTrip


# ── Median binning, DB-side (mirrors training_forecast_models.py) ────────────
class Median(Aggregate):
    function = "PERCENTILE_CONT"
    name = "median"
    output_field = FloatField()
    template = "%(function)s(0.5) WITHIN GROUP (ORDER BY %(expressions)s)"
    allow_distinct = False


def _bin_expr(dk):
    # date_bin([dk] seconds, ts, 1970-01-01) → same edges as pandas .floor()
    return Func(
        F("tpep_pickup_datetime"),
        function="date_bin",
        template=("date_bin(INTERVAL '%(dk)s seconds', %(expressions)s, "
                  "TIMESTAMP '1970-01-01 00:00:00')"),
        dk=dk,
        output_field=DateTimeField(),
    )


METRICS = [
    "sefer", "passenger_count", "trip_distance", "fare_amount",
    "tip_amount", "tolls_amount", "total_amount",
    "trip_duration_minutes", "trip_speed_mph", "price_per_distance",
    "hourly_trip_volume", "hourly_avg_speed",
]
AVG_METRICS = METRICS[1:]                      # everything except the row-count "sefer"

RES_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "10m": 600, "15m": 900, "30m": 1800,
    "1h": 3600, "3h": 10800, "1d": 86400,
}


def _binned_long(vendor, resolution):
    """One (vendor, resolution) → long DataFrame of every metric's median (no split yet)."""
    dk = RES_SECONDS[resolution]
    qs = TrainTrip.objects.all()
    if vendor != "hepsi":
        qs = qs.filter(vendorid=vendor)

    agg = {"sefer": Count("id")}
    for m in AVG_METRICS:
        agg[f"m_{m}"] = Median(m)

    rows = (qs.annotate(_bin=_bin_expr(dk))
              .values("_bin")
              .annotate(**agg)
              .order_by("_bin"))
    df = pd.DataFrame(list(rows))
    if len(df) == 0:
        return pd.DataFrame(columns=["series_id", "metric", "resolution",
                                     "vendor", "ts", "value"])

    df = df.rename(columns={f"m_{m}": m for m in AVG_METRICS})
    df["_bin"] = pd.to_datetime(df["_bin"])

    parts = []
    for metric in METRICS:
        sub = df[["_bin", metric]].dropna(subset=[metric]).copy()
        if len(sub) == 0:
            continue
        sub = sub.rename(columns={"_bin": "ts", metric: "value"})
        sub["series_id"]  = f"{metric}__{resolution}__{vendor}"
        sub["metric"]     = metric
        sub["resolution"] = resolution
        sub["vendor"]     = vendor
        parts.append(sub[["series_id", "metric", "resolution", "vendor", "ts", "value"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["series_id", "metric", "resolution", "vendor", "ts", "value"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="taxi_series.parquet")
    ap.add_argument("--resolutions", default=",".join(RES_SECONDS))
    ap.add_argument("--vendors", default="hepsi,1,2")
    ap.add_argument("--test-days", type=int, default=120,
                    help="last N days of 2015-16 → val holdout (2017 ignored)")
    args = ap.parse_args()

    resolutions = [r.strip() for r in args.resolutions.split(",") if r.strip()]
    vendors     = [v.strip() for v in args.vendors.split(",") if v.strip()]

    all_parts = []
    for vendor in vendors:
        for resolution in resolutions:
            print(f"  vendor={vendor} res={resolution} ...", end=" ", flush=True)
            part = _binned_long(vendor, resolution)
            print(f"{len(part):,} rows, {part['series_id'].nunique()} series")
            if len(part):
                all_parts.append(part)

    out = pd.concat(all_parts, ignore_index=True)

    # time-based split: last --test-days → val, rest → train (global cutoff)
    cutoff = out["ts"].max().normalize() - timedelta(days=args.test_days)
    out["split"] = out["ts"].ge(cutoff).map({True: "val", False: "train"})
    out = out[["series_id", "metric", "resolution", "vendor", "split", "ts", "value"]]
    out = out.sort_values(["series_id", "split", "ts"]).reset_index(drop=True)
    out.to_parquet(args.out, index=False)

    print(f"\ncutoff (val starts): {cutoff.date()}")
    print(f"split rows: {out['split'].value_counts().to_dict()}")
    print(f"DONE → {args.out}")
    print(f"  total rows : {len(out):,}")
    print(f"  series     : {out['series_id'].nunique()}")
    print(f"  splits     : {out.groupby('split')['series_id'].nunique().to_dict()}")
    print(f"  date range : {out['ts'].min()} → {out['ts'].max()}")


if __name__ == "__main__":
    main()