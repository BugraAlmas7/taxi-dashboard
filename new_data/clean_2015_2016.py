"""
clean_2015_2016.py
──────────────────
Standalone cleaning tool for the 2015-16 NYC taxi trips. It:

  1. MERGES many monthly parquet files (or reads a single parquet) into one stream,
  2. engineers the trip features (duration, speed, price-per-distance),
  3. FLAGS anomalies at the raw-trip level with an Isolation Forest,
  4. writes the results, and — only if you ask for it with --drop — DELETES the
     flagged rows and writes a cleaned file.

It does NOT touch the database and needs no Django. Pure polars + pyarrow +
scikit-learn, streamed in batches so it stays memory-safe on multi-GB inputs.

Outputs (in --out-dir):
    flagged.parquet     every row + `is_anomaly` (0/1) + `anomaly_score`
    anomalies.parquet   only the flagged rows (for review before deleting)
    cleaned.parquet     flagged rows removed  (written ONLY with --drop)

Examples:
    # just flag, review anomalies.parquet first, delete nothing
    python clean_2015_2016.py --input ./raw_2015_2016 --out-dir ./data

    # flag AND write a cleaned file with the flagged rows removed
    python clean_2015_2016.py --input ./raw_2015_2016 --out-dir ./data --drop

    # a single already-merged file, stricter flagging
    python clean_2015_2016.py --input ./data/merged_2015_2016.parquet \
        --out-dir ./data --contamination 0.03 --drop
"""
import os
import glob
import argparse

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import IsolationForest

# Raw-trip features the Isolation Forest scores on (same set the live pipeline
# cleans on). Negative speeds, absurd distances/fares fall out here.
FEATURES = ["trip_distance", "trip_duration_minutes", "trip_speed_mph",
            "price_per_distance"]


def _sources(input_path):
    """Return the list of parquet files to read (a folder → every *.parquet)."""
    if os.path.isdir(input_path):
        files = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
        if not files:
            raise SystemExit(f"No .parquet files found in folder: {input_path}")
        return files
    if not os.path.exists(input_path):
        raise SystemExit(f"Input not found: {input_path}")
    return [input_path]


def _engineer(df: pl.DataFrame) -> pl.DataFrame:
    """Lowercase columns and derive duration / speed / price-per-distance."""
    df = df.rename({c: c.lower() for c in df.columns})
    df = df.with_columns(
        ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
         .dt.total_seconds() / 60.0).alias("trip_duration_minutes")
    )
    df = df.with_columns(
        pl.when(pl.col("trip_duration_minutes") > 0)
          .then(pl.col("trip_distance") / (pl.col("trip_duration_minutes") / 60.0))
          .otherwise(None).alias("trip_speed_mph")
    )
    df = df.with_columns(
        pl.when(pl.col("trip_distance") > 0)
          .then(pl.col("total_amount") / pl.col("trip_distance"))
          .otherwise(None).alias("price_per_distance")
    )
    return df


def _feature_matrix(df: pl.DataFrame) -> np.ndarray:
    """Build the (n, 4) float32 feature matrix; inf/NaN → 0 so the model can score
    every row (rows with invalid features tend to be flagged as anomalies)."""
    X = df.select(FEATURES).to_numpy().astype(np.float32)
    X[~np.isfinite(X)] = 0.0
    return X


def fit_model(files, contamination, fit_sample, batch_rows, seed):
    """Pass 1 — collect a random sample of feature rows and fit the Isolation Forest."""
    rng = np.random.default_rng(seed)
    pool, collected = [], 0
    print(f"[fit] sampling up to {fit_sample:,} rows to fit Isolation Forest...")
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_rows):
            X = _feature_matrix(_engineer(pl.from_arrow(batch)))
            take = min(len(X), max(1, fit_sample // 20))
            idx = rng.choice(len(X), take, replace=False) if len(X) > take else np.arange(len(X))
            pool.append(X[idx])
            collected += len(idx)
            if collected >= fit_sample:
                break
        if collected >= fit_sample:
            break
    Xfit = np.vstack(pool)[:fit_sample]
    print(f"[fit] fitting on {len(Xfit):,} rows (contamination={contamination})...")
    model = IsolationForest(n_estimators=100, contamination=contamination,
                            random_state=seed, n_jobs=-1)
    model.fit(Xfit)
    return model


def apply_model(files, model, out_dir, batch_rows, drop):
    """Pass 2 — score every row, tag it, and stream the outputs to disk."""
    flagged_path   = os.path.join(out_dir, "flagged.parquet")
    anomalies_path = os.path.join(out_dir, "anomalies.parquet")
    cleaned_path   = os.path.join(out_dir, "cleaned.parquet")

    w_flagged = w_anom = w_clean = None
    n_total = n_anom = 0
    print("[apply] scoring and writing outputs...")
    try:
        for path in files:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=batch_rows):
                df = _engineer(pl.from_arrow(batch))
                X = _feature_matrix(df)
                pred = model.predict(X)                       # -1 anomaly, 1 normal
                score = -model.decision_function(X)           # higher = more anomalous
                is_anom = (pred == -1)

                df = df.with_columns([
                    pl.Series("anomaly_score", score.astype(np.float32)),
                    pl.Series("is_anomaly", is_anom.astype(np.int8)),
                ])
                tbl = df.to_arrow()

                if w_flagged is None:            # lock the schema from the first batch
                    w_flagged = pq.ParquetWriter(flagged_path, tbl.schema)
                    w_anom    = pq.ParquetWriter(anomalies_path, tbl.schema)
                    if drop:
                        w_clean = pq.ParquetWriter(cleaned_path, tbl.schema)

                w_flagged.write_table(tbl)
                anom_tbl = df.filter(pl.col("is_anomaly") == 1).to_arrow()
                if anom_tbl.num_rows:
                    w_anom.write_table(anom_tbl)
                if drop:
                    clean_tbl = df.filter(pl.col("is_anomaly") == 0).to_arrow()
                    if clean_tbl.num_rows:
                        w_clean.write_table(clean_tbl)

                n_total += len(df)
                n_anom  += int(is_anom.sum())
                print(f"    processed {n_total:,} rows  |  flagged {n_anom:,}", end="\r")
    finally:
        for w in (w_flagged, w_anom, w_clean):
            if w is not None:
                w.close()

    print()
    rate = (n_anom / n_total * 100) if n_total else 0.0
    print(f"[done] total={n_total:,}  flagged={n_anom:,} ({rate:.2f}%)  clean={n_total - n_anom:,}")
    print(f"       flagged   → {flagged_path}")
    print(f"       anomalies → {anomalies_path}")
    if drop:
        print(f"       cleaned   → {cleaned_path}  (flagged rows removed)")
    else:
        print("       (no rows deleted — re-run with --drop to write cleaned.parquet)")


def main():
    ap = argparse.ArgumentParser(description="Merge, flag and optionally drop anomalous 2015-16 taxi trips.")
    ap.add_argument("--input", required=True,
                    help="folder of monthly *.parquet files, or a single parquet file")
    ap.add_argument("--out-dir", default=".", help="where to write the output parquet files")
    ap.add_argument("--contamination", type=float, default=0.02,
                    help="expected anomaly fraction for Isolation Forest (default 0.02)")
    ap.add_argument("--fit-sample", type=int, default=1_000_000,
                    help="rows sampled to fit the model (default 1,000,000)")
    ap.add_argument("--batch-rows", type=int, default=500_000,
                    help="rows read per streaming batch (default 500,000)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop", action="store_true",
                    help="also write cleaned.parquet with the flagged rows removed")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = _sources(args.input)
    print(f"Input files ({len(files)}):")
    for f in files:
        print(f"  - {f}")

    model = fit_model(files, args.contamination, args.fit_sample, args.batch_rows, args.seed)
    apply_model(files, model, args.out_dir, args.batch_rows, args.drop)


if __name__ == "__main__":
    main()
