# NYC Taxi — Streaming Anomaly Cleaning & Live Forecasting

A Django dashboard for NYC Yellow Taxi demand/revenue forecasting with a **live
streaming pipeline**: it replays a trip dataset as a stream, cleans each window
with an **Isolation Forest**, incrementally updates forecast models
(**XGBoost** + **Chronos-2**) on the cleaned data, and shows the forecast
running *ahead* of the actuals — as if forecasting in real time. Every simulated
midnight the pipeline cleans the finished day and retrains/fine-tunes the live
models.

> The simulated clock is driven from the dataset timestamps (starting at a
> configurable `start`), **not** the wall clock, so the demo is reproducible and
> timezone-independent.

## What it does

- **Simulate** a stream from a trip table in time order (`streaming/simulator.py`).
- **Clean** each window's raw trips with a fresh Isolation Forest
  (`streaming/cleaning.py`) — negative speeds, absurd distances/fares removed.
- **Update** forecast models on the cleaned, binned series
  (`streaming/updaters.py`):
  - *XGBoost* — true incremental `init_model` fine-tune (or fresh train).
  - *Chronos-2* — periodic LoRA step, hot-swapped into the live panel.
- **Serve** a live 5-panel page where the forecast leads the actuals, plus a
  classic dashboard comparing baseline vs. live models.

Full design notes: [`anomaly/streaming_pipeline_architecture.md`](anomaly/streaming_pipeline_architecture.md).

## Tech stack

Django 6 · PostgreSQL · pandas · scikit-learn (Isolation Forest) · XGBoost ·
Chronos-2 / TimesFM (Hugging Face + PEFT/LoRA) · Plotly · Docker.

## Prerequisites

- **Docker** + Docker Compose. That's it — a **PostgreSQL 18** instance is
  bundled as the `db` service in `docker-compose.yml`, so no external database is
  required. (You can still point at an external Postgres via `.env`; see Notes.)
- Internet access for the **image build** (pip installs), and for the **first**
  Chronos run (downloads `amazon/chronos-2` from Hugging Face). The XGBoost live
  demo runs fully offline once models exist.
- Large **data** and **trained models** are **not** in the repo (see
  `.gitignore`); you load the data and generate the models yourself (below).

## Setup

1. **Clone & enter**

   ```bash
   git clone https://github.com/BugraAlmas7/taxi-dashboard taxi && cd taxi
   ```

2. **Build & run — one command:**

   ```bash
   docker compose up --build
   ```

   This starts the bundled Postgres **and** the app. On startup the container's
   entrypoint automatically creates every table (`bootstrap_db`), so there is no
   manual `migrate` step. No `.env` is needed — compose has sensible defaults.
   Open <http://localhost:8000/>. The pages render immediately; charts are empty
   until you load data (next step).

   > Why not plain `migrate`? The trip tables (`sefer_egitim`, `sefer_2017`) and
   > the profile/cache tables are `managed = False` (historically created outside
   > Django) and the migration chain tries to `ALTER` them, which crashes on a
   > fresh DB. The entrypoint runs `bootstrap_db`, which fake-applies those
   > migrations, creates every table with the schema editor, and adds any column
   > the model gained after the table was first created (idempotent).

3. **Load the trip data.** Two ways:

   - **Automatic (once):** create your `.env` first — `cp .env.example .env` —
     then in it set `RUN_SETUP_DATA=1` (and optionally
     `SETUP_DATA_ARGS=--months-train 2016-12:2016-12 --months-test 2017-01:2017-01`
     (Dec 2016 is required for the live-page prewarm — the 30 days before the 2017-01-01 demo start)
     to limit the download), then `docker compose up`. The entrypoint downloads +
     cleans + loads on first boot and skips if data already exists.
   - **Manual:** `docker compose exec web python setup_data.py` (add
     `--months-*` to limit, `--no-download` to reuse files already in `data/`).

   It downloads the raw monthly parquet from the official NYC TLC bucket, MAD-cleans
   2015-16, keeps 2017 raw, and loads both into Postgres. The full dataset is tens
   of GB — start with a small month subset. (The bundled DB is also exposed on host
   port **5433** for loading with external tools if you prefer.)

4. **Create an account.** The pages require a login: open
   <http://localhost:8000/register/>, sign up, then log in. (Plain Django auth —
   no email verification; the first account is yours.)

5. **Generate the models** *(optional for the live demo — the live pipeline
   trains its own models from the loaded data as it runs, writing them into
   `models_update/xgboost_stream/` and `finetune/chronos_ft_stream/`)*.
   The **baseline dashboard models** are not shipped and are regenerated
   locally, inside the running container:

   ```bash
   docker compose exec web python training/training_anomaly_models.py
   docker compose exec web python training/training_forecast_models.py   # --help for options
   ```

   The fine-tuned foundation models (Chronos-2 / TimesFM LoRA) come from the
   Colab notebooks in `notebooks/` — run them on Colab (GPU) and copy the
   checkpoints into `finetune/` as described in `finetune/README.md` and
   `models_update/README.md`.

6. Open <http://localhost:8000/> (dashboard) and <http://localhost:8000/live/>
   (live forecast). Use `localhost`, **not** `0.0.0.0`.

## Multi-container architecture (web · worker · redis · db)

The stack is split by JOB, not by screen — four containers on one compose network:

| Container | Role |
|---|---|
| **web** | Django UI (all screens) + API. Enqueues heavy jobs, never runs them. |
| **worker** | Celery worker. Runs the streaming pipeline (IF cleaning + XGBoost/Chronos updates) OFF the UI thread. |
| **redis** | Broker/queue between web and worker. |
| **db** | PostgreSQL (bundled, or an external one via `.env`). |

`web` and `worker` share the same image and the same `.:/app` mount, so the model
files the worker writes (`models_update/`, `finetune/`) are exactly the ones the
web reads. Nothing is handed off in memory — the seam is the shared DB + Redis:

```
web  ── POST /pipeline/start ──▶ redis ──▶ worker runs Orchestrator loop
                                            │ writes PipelineWindowResult rows (DB)
web  ◀── GET /pipeline/status ──────────────┘ (polls those rows)
web  ── POST /pipeline/stop  ──▶ redis flag ▶ worker stops between windows
```

Endpoints (login required): `POST /pipeline/start` (optional form fields:
`models`, `mode`, `source`, `start`, `max_windows`, `tick_sleep`, `run_name`),
`POST /pipeline/stop`, `GET /pipeline/status`. `docker compose up --build` starts
all four; scale the heavy side with `docker compose up --scale worker=2`.

## Using the live page

Pick a model (**XGBoost (Live)** is fast and CPU-friendly), a speed, and press
**Start**. The clock advances hour by hour; at each simulated midnight the page
calls the retrain endpoint, which cleans that day and updates the live model
(the first retrain also warms up on ~30 days of history, so it takes a few
minutes — this is normal). **Chronos (Live)** is much slower on CPU and needs
the Hugging Face download on first use.

## Notes & gotchas

- **Using an external Postgres instead of the bundled `db`.** Set `POSTGRES_HOST`
  (and the other `POSTGRES_*`) in `.env`. If Postgres runs on Windows and Docker
  runs in WSL, the container reaches it via the Windows host IP — the `ip route`
  default gateway (e.g. `172.29.128.1`), **not** `host.docker.internal` (which
  resolves to the WSL VM). Two things bite after a reboot: the Postgres service
  may not auto-start (set it to *Automatic*), and **Windows Firewall** may block
  port 5432 from WSL — add an inbound rule:

  ```powershell
  New-NetFirewallRule -DisplayName "PostgreSQL 5432 WSL" -Direction Inbound -LocalPort 5432 -Protocol TCP -Action Allow
  ```

  Test from the WSL shell: `bash -c "cat < /dev/null > /dev/tcp/<host>/5432" && echo OPEN`.

- **Timezone.** `USE_TZ = False` with `TIME_ZONE = America/New_York`; the
  container pins `TZ=America/New_York` (Dockerfile + compose) so behaviour matches
  across machines. Do **not** add `TIME_ZONE` under `DATABASES` — Django refuses
  it while `USE_TZ = False`.

- **CPU vs GPU.** Everything runs on CPU. XGBoost is cheap; Chronos LoRA
  fine-tuning is slow on CPU. For GPU, pass devices to the container
  (`deploy.resources.reservations.devices` + NVIDIA Container Toolkit) — not
  required for the XGBoost demo.

- **First-run gotchas.** Table creation is automatic (the entrypoint runs
  `bootstrap_db`); you never call plain `migrate`. Postgres only applies
  `POSTGRES_PASSWORD` when the data volume is **first** created — if you change DB
  credentials later, reset the volume with `docker compose down -v` (this wipes the
  bundled DB) before `up`.

- **Not committed:** `data/`, `models_update/`, `finetune/` checkpoints and
  `*.parquet`, `venv/`, `db.sqlite3`, and `.env`. See `.gitignore`.

## Project layout

```
taxi/
├── anomaly/                 # main Django app
│   ├── streaming/           # simulator, sliding window, cleaning, updaters, orchestrator
│   ├── models.py            # trip tables + PipelineWindowResult
│   ├── forecast_calculation.py
│   ├── views.py / views_live.py
│   └── templates/anomaly/live.html
├── taxi_web/                # Django project (settings, urls, wsgi/asgi)
├── training/                # baseline + forecast model training scripts
├── finetune/                # LoRA adapters land here (gitignored) + export_series.py
├── notebooks/               # Colab fine-tuning notebooks (Chronos-2 / TimesFM)
├── new_data/                # data import / indexing + clean_2015_2016.py
├── models_update/           # trained model weights land here (gitignored)
├── data/                    # datasets land here (gitignored)
├── Dockerfile · docker-compose.yml · requirements.txt
└── .env.example
```
