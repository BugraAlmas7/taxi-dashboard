"""
setup_data.py
─────────────
One-shot data bootstrap for the NYC Taxi project. It:

  1. DOWNLOADS the raw monthly parquet files from the official NYC TLC bucket
     (2015-01 … 2016-12 for training, 2017-01 … 2017-12 for the test stream),
  2. ENGINEERS features (duration, speed, price-per-distance, hourly aggregates),
     MAD-cleans the 2015-16 training set and keeps 2017 raw,
  3. WRITES the result into PostgreSQL via the Django ORM
     (`sefer_egitim` = TrainTrip, `sefer_2017` = Trip2017).

Run it inside the web container (so Django + the DB are configured):

    docker compose exec web python setup_data.py                 # download + clean + load
    docker compose exec web python setup_data.py --no-download   # use files already in data/
    docker compose exec web python setup_data.py --months-train 2016-01:2016-12 \
        --months-test 2017-01:2017-03                            # smaller subset for a quick test

NOTE: the full 2015-16 + 2017 raw data is tens of GB. Use --months-* to test with
fewer months first. Files that already exist are not re-downloaded.
"""
import os
import sys
import glob
import argparse
import urllib.request

import polars as pl
import pyarrow.parquet as pq

# ── Django setup (must happen before importing models) ───────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
import django
django.setup()

from django.db import connection
from anomaly.models import TrainTrip, Trip2017

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR       = os.path.join(BASE_DIR, "data")
RAW_TRAIN_DIR  = os.path.join(DATA_DIR, "raw_2015_2016")
RAW_TEST_DIR   = os.path.join(DATA_DIR, "raw_2017")
CLEANED_TRAIN  = os.path.join(DATA_DIR, "cleaned_2015_2016.parquet")
PURE_TEST      = os.path.join(DATA_DIR, "pure_2017.parquet")

# Official NYC TLC parquet bucket (yellow taxi)
TLC_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{ym}.parquet"

# Columns that exist on the Django models (anything else in the raw file is dropped).
# Older 2015/early-2016 files have lng/lat instead of pulocationid/dolocationid — the
# available-columns filter below handles that automatically.
MODEL_COLUMNS = [
    "vendorid", "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "passenger_count", "trip_distance", "fare_amount", "tip_amount",
    "tolls_amount", "total_amount", "pulocationid", "dolocationid",
    "trip_duration_minutes", "trip_speed_mph", "price_per_distance",
    "hourly_trip_volume", "hourly_avg_speed",
]

TARGETS = [(TrainTrip, CLEANED_TRAIN), (Trip2017, PURE_TEST)]


# ── month-range helpers ──────────────────────────────────────────────────────
def month_range(spec):
    """'2015-01:2016-12' → ['2015-01', ..., '2016-12']."""
    lo, hi = spec.split(":")
    ly, lm = map(int, lo.split("-"))
    hy, hm = map(int, hi.split("-"))
    out, y, m = [], ly, lm
    while (y, m) <= (hy, hm):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def download_months(months, dst_dir, label):
    os.makedirs(dst_dir, exist_ok=True)
    print(f"[{label}] downloading {len(months)} monthly file(s) → {dst_dir}")
    for ym in months:
        out = os.path.join(dst_dir, f"yellow_tripdata_{ym}.parquet")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            print(f"    {ym}: already present, skip")
            continue
        url = TLC_URL.format(ym=ym)
        try:
            print(f"    {ym}: downloading ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, out + ".part")
            os.replace(out + ".part", out)
            print(f"{os.path.getsize(out) / 1e6:.0f} MB")
        except Exception as e:
            if os.path.exists(out + ".part"):
                os.remove(out + ".part")
            print(f"FAILED ({type(e).__name__}: {e}) — skipping {ym}")


# ── feature engineering + cleaning ───────────────────────────────────────────
def _mad_filter(lf, column, k=3.0):
    """Median-Absolute-Deviation outlier filter: keep |0.6745*(x-med)/MAD| <= k."""
    med = pl.col(column).median()
    mad = (pl.col(column) - med).abs().median()
    safe_mad = pl.when(mad == 0).then(1e-9).otherwise(mad)
    z = 0.6745 * (pl.col(column) - med) / safe_mad
    return lf.filter(z.abs() <= k)


def engineer(input_dir, output_file, label, clean_anomalies):
    files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not files:
        print(f"[{label}] SKIPPED: no .parquet files in {input_dir}")
        return False

    print(f"\n[{label}] engineering from {len(files)} file(s) ...")
    lf = pl.scan_parquet(files)
    names = lf.collect_schema().names()
    lf = lf.rename({c: c.lower() for c in names})

    lf = lf.with_columns(
        ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
         .dt.total_seconds() / 60.0).alias("trip_duration_minutes")
    ).with_columns(
        pl.when(pl.col("trip_duration_minutes") > 0)
          .then(pl.col("trip_distance") / (pl.col("trip_duration_minutes") / 60.0))
          .otherwise(None).alias("trip_speed_mph"),
        # price-per-distance uses TOTAL_AMOUNT to match the trained models / the rest
        # of the pipeline (data_prep.py, streaming). (Gemini used fare_amount here.)
        pl.when(pl.col("trip_distance") > 0)
          .then(pl.col("total_amount") / pl.col("trip_distance"))
          .otherwise(None).alias("price_per_distance"),
    )

    # drop fundamental data-entry errors
    lf = lf.filter(
        (pl.col("trip_duration_minutes") > 0)
        & (pl.col("trip_distance") >= 0)
        & (pl.col("total_amount") >= 0)
    )

    # MAD cleaning only for the training set (keep 2017 raw)
    if clean_anomalies:
        print(f"[{label}] MAD outlier removal (k=3.0) ...")
        for col in ("trip_speed_mph", "trip_duration_minutes", "price_per_distance"):
            lf = _mad_filter(lf, col, k=3.0)

    # hourly aggregates (same value repeated for every trip in the hour)
    lf = lf.with_columns(
        pl.col("tpep_pickup_datetime").dt.truncate("1h").alias("pickup_hour")
    ).with_columns(
        pl.len().over("pickup_hour").cast(pl.Int32).alias("hourly_trip_volume"),
        pl.col("trip_speed_mph").mean().over("pickup_hour").alias("hourly_avg_speed"),
    )

    available = lf.collect_schema().names()
    final_cols = [c for c in MODEL_COLUMNS if c in available]
    lf.select(final_cols).sink_parquet(output_file)
    print(f"[{label}] saved → {output_file}")
    return True


# ── DB load (Django ORM) ─────────────────────────────────────────────────────
def ensure_table(model):
    """Create the table for an unmanaged model if it does not exist yet
    (managed=False models are NOT created by `migrate`)."""
    if model._meta.managed:
        return
    table = model._meta.db_table
    with connection.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", [table])
        exists = cur.fetchone()[0] is not None
    if not exists:
        with connection.schema_editor() as se:
            se.create_model(model)
        print(f"    created missing table '{table}'")


def bulk_insert(model, parquet_path):
    print(f"\n[{model.__name__}] loading {parquet_path}")
    ensure_table(model)
    field_names = {f.name for f in model._meta.fields}
    pf = pq.ParquetFile(parquet_path)

    model.objects.all().delete()
    print(f"[{model.__name__}] cleared existing rows")

    total = 0
    for batch in pf.iter_batches(batch_size=50_000):
        df = pl.from_arrow(batch)
        rows = df.to_dicts()
        # keep only keys the model actually has (robust to extra parquet columns)
        objs = [model(**{k: v for k, v in r.items() if k in field_names}) for r in rows]
        model.objects.bulk_create(objs, batch_size=5000, ignore_conflicts=True)
        total += len(objs)
        print(f"    +{len(objs):,}  (total {total:,})", end="\r")
    print(f"\n[{model.__name__}] done: {total:,} rows")


def main():
    ap = argparse.ArgumentParser(description="Download, clean and load NYC taxi data into Postgres.")
    ap.add_argument("--no-download", action="store_true",
                    help="skip downloading; use whatever parquet files are already in data/")
    ap.add_argument("--months-train", default="2015-01:2016-12",
                    help="training month range YYYY-MM:YYYY-MM (default 2015-01:2016-12)")
    ap.add_argument("--months-test", default="2017-01:2017-12",
                    help="test month range YYYY-MM:YYYY-MM (default 2017-01:2017-12)")
    ap.add_argument("--skip-engineer", action="store_true",
                    help="skip step 1 (reuse existing cleaned_/pure_ parquet files)")
    ap.add_argument("--skip-load", action="store_true",
                    help="skip step 2 (only produce the parquet files, don't touch the DB)")
    ap.add_argument("--if-empty", action="store_true",
                    help="do nothing if the trip tables already contain rows (used by the entrypoint)")
    args = ap.parse_args()

    if args.if_empty:
        try:
            if Trip2017.objects.exists() or TrainTrip.objects.exists():
                print("[setup_data] tables already have data — nothing to do (--if-empty).")
                return
        except Exception:
            pass  # tables may not exist yet → proceed

    os.makedirs(RAW_TRAIN_DIR, exist_ok=True)
    os.makedirs(RAW_TEST_DIR, exist_ok=True)

    if not args.no_download:
        print("=== STEP 0: download raw parquet from NYC TLC ===")
        download_months(month_range(args.months_train), RAW_TRAIN_DIR, "train 2015-16")
        download_months(month_range(args.months_test),  RAW_TEST_DIR,  "test 2017")

    if not args.skip_engineer:
        print("\n=== STEP 1: feature engineering + cleaning ===")
        engineer(RAW_TRAIN_DIR, CLEANED_TRAIN, "train 2015-16", clean_anomalies=True)
        engineer(RAW_TEST_DIR,  PURE_TEST,     "test 2017 (raw)", clean_anomalies=False)

    if not args.skip_load:
        print("\n=== STEP 2: load into PostgreSQL (Django ORM) ===")
        for model, path in TARGETS:
            if not os.path.exists(path):
                print(f"[{model.__name__}] SKIPPED: {path} not found")
                continue
            bulk_insert(model, path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
