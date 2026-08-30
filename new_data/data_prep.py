import polars as pl
import pyarrow.parquet as pq
import os

file_path = "data/cleaned_2015_2016.parquet"
output_path = "D:/data/engineered_2015_2016.csv"

print("Step 1: Computing the hourly-volume summary table...")

df_agg = (
    pl.scan_parquet(file_path)
    .select(["tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance"])
    .with_columns([
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        pl.col("tpep_pickup_datetime").dt.date().alias("pickup_date"),
        ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds() / 60).alias("trip_duration_minutes"),
    ])
    .with_columns([
        (pl.col("trip_distance") / pl.when(pl.col("trip_duration_minutes") > 0)
         .then(pl.col("trip_duration_minutes") / 60)
         .otherwise(0.001)).alias("trip_speed_mph"),
    ])
    .group_by(["pickup_date", "pickup_hour"])
    .agg([
        pl.len().alias("hourly_trip_volume"),
        pl.col("trip_speed_mph").mean().alias("hourly_avg_speed")
    ])
    .collect()
)
print("Step 1 done. Summary table ready.")


print("Step 2: Processing the main data and writing to CSV chunk by chunk...")

parquet_file = pq.ParquetFile(file_path)
is_first_chunk = True

for i in range(parquet_file.num_row_groups):
    print(f"Processing chunk: {i+1} / {parquet_file.num_row_groups}")

    chunk_df = pl.from_arrow(parquet_file.read_row_group(i))

    chunk_engineered = chunk_df.with_columns([
        pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
        pl.col("tpep_pickup_datetime").dt.date().alias("pickup_date"),
        ((pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime")).dt.total_seconds() / 60).alias("trip_duration_minutes"),
    ]).with_columns([
        (pl.col("trip_distance") / pl.when(pl.col("trip_duration_minutes") > 0)
         .then(pl.col("trip_duration_minutes") / 60)
         .otherwise(0.001)).alias("trip_speed_mph"),
    ]).with_columns([
        (pl.col("total_amount") / pl.when(pl.col("trip_distance") > 0)
         .then(pl.col("trip_distance"))
         .otherwise(0.001)).alias("price_per_distance")
    ])


    final_chunk = chunk_engineered.join(df_agg, on=["pickup_date", "pickup_hour"], how="left")


    mode = "a" if not is_first_chunk else "w"
    with open(output_path, mode=mode, encoding="utf-8") as f:
        final_chunk.write_csv(f, include_header=is_first_chunk)

    is_first_chunk = False

print("ALL DONE! The machine survived :)")
