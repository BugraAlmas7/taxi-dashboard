# data/ — datasets (NOT committed)

Large parquet/CSV files live here and are **gitignored**. Expected files:

- `cleaned_2015_2016.parquet` — cleaned 2015-16 trips (training source)
- `pure_2017.parquet` — 2017 trips (the streamed test set)
- outputs of `../new_data/clean_2015_2016.py`: `flagged.parquet`,
  `anomalies.parquet`, `cleaned.parquet`

These feed the Postgres tables the app reads (see the project README).
