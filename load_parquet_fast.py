"""
load_parquet_fast.py
────────────────────
FAST bulk load of the pre-engineered parquet files into Postgres using the
native COPY protocol — 10-20x faster than setup_data.py's ORM bulk_create.

Use this when data/cleaned_2015_2016.parquet and data/pure_2017.parquet ALREADY
exist (e.g. restored from a backup). It does NO download and NO feature
engineering — it never touches the parquet files, only reads them.

    docker compose exec web python load_parquet_fast.py              # both tables
    docker compose exec web python load_parquet_fast.py --only train # sefer_egitim only
    docker compose exec web python load_parquet_fast.py --only test  # sefer_2017 only
    docker compose exec web python load_parquet_fast.py --keep       # append instead of truncate

Each table is TRUNCATEd first (unless --keep), then loaded in 200k-row chunks
streamed through COPY ... FROM STDIN (CSV). Columns present in the parquet but
not on the Django model are dropped; model columns missing from the parquet
stay NULL — same tolerant behaviour as setup_data.bulk_insert.
"""
import io
import os
import sys
import time
import argparse

import polars as pl
import pyarrow.parquet as pq

# ── Django setup ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
import django
django.setup()

from django.db import connection
from anomaly.models import TrainTrip, Trip2017

DATA_DIR = os.path.join(BASE_DIR, "data")
TARGETS = {
    "train": (TrainTrip, os.path.join(DATA_DIR, "cleaned_2015_2016.parquet")),
    "test":  (Trip2017,  os.path.join(DATA_DIR, "pure_2017.parquet")),
}
BATCH = 200_000


def load(model, path, keep=False):
    table = model._meta.db_table
    if not os.path.exists(path):
        print(f"[{table}] SKIPPED: {path} not found")
        return

    # columns = intersection of parquet columns and model columns (id excluded)
    model_cols = {f.column for f in model._meta.fields} - {"id"}
    pf = pq.ParquetFile(path)
    cols = [c for c in pf.schema_arrow.names if c.lower() in model_cols]
    if not cols:
        print(f"[{table}] SKIPPED: no matching columns in {path}")
        return

    n_total = pf.metadata.num_rows
    print(f"[{table}] loading {n_total:,} rows from {os.path.basename(path)}")
    print(f"[{table}] columns: {', '.join(cols)}")

    # ensure the table exists (same trick as setup_data.ensure_table)
    with connection.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", [table])
        if cur.fetchone()[0] is None:
            with connection.schema_editor() as se:
                se.create_model(model)
            print(f"[{table}] created missing table")

    with connection.cursor() as cur:
        if not keep:
            cur.execute(f'TRUNCATE TABLE "{table}"')
            print(f"[{table}] truncated")

    raw_conn = connection.connection            # underlying psycopg3 connection
    col_list = ", ".join(f'"{c.lower()}"' for c in cols)
    copy_sql = f'COPY "{table}" ({col_list}) FROM STDIN WITH (FORMAT csv)'

    t0 = time.time()
    total = 0
    for batch in pf.iter_batches(batch_size=BATCH, columns=cols):
        df = pl.from_arrow(batch)
        buf = io.BytesIO()
        df.write_csv(buf, include_header=False)
        data = buf.getvalue()
        with raw_conn.cursor() as cur:
            with cur.copy(copy_sql) as cp:
                cp.write(data)
        total += len(df)
        rate = total / max(time.time() - t0, 1e-9)
        eta = (n_total - total) / max(rate, 1)
        print(f"  {total:,}/{n_total:,}  ({rate:,.0f} rows/s · ETA {eta/60:,.1f} min)",
              end="\r", flush=True)
    print(f"\n[{table}] done: {total:,} rows in {(time.time()-t0)/60:.1f} min")


def main():
    ap = argparse.ArgumentParser(description="Fast COPY-based parquet → Postgres loader.")
    ap.add_argument("--only", choices=["train", "test"],
                    help="load just one table (default: both)")
    ap.add_argument("--keep", action="store_true",
                    help="append to existing rows instead of TRUNCATE first")
    args = ap.parse_args()

    picks = [args.only] if args.only else ["train", "test"]
    for key in picks:
        model, path = TARGETS[key]
        load(model, path, keep=args.keep)
    print("\nAll done.")


if __name__ == "__main__":
    main()
