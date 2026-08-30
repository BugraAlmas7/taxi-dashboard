# finetune/ — foundation-model fine-tuning (weights NOT committed)

LoRA adapters are large and **regenerated**; only the structure is committed.
The fine-tuning notebooks are under `../notebooks/`.

| Folder                | Produced by                         | What it holds                              |
|-----------------------|-------------------------------------|--------------------------------------------|
| `chronos_ft/`         | `notebooks/chronos_ft.ipynb`        | Chronos-2 LoRA adapter                      |
| `chronos_ft_optuna/`  | `notebooks/chronos_ft_optuna.ipynb` + `ch_ft_optuna.ipynb` | Optuna-tuned Chronos-2 adapter |
| `chronos_ft_stream/`  | live streaming pipeline             | per-metric Chronos-2 adapters updated live |
| `timesfm_ft/`         | `notebooks/timesfm_ft.ipynb`        | TimesFM-2.5 LoRA adapter                    |
| `timesfm_ft_optuna/`  | `notebooks/timesfm_ft_optuna.ipynb` + `tfm_ft_optuna.ipynb` | Optuna-tuned TimesFM adapter |

`export_series.py` (and `notebooks/data_for_finetune.ipynb`) build the
`taxi_series.parquet` used as the fine-tuning input.
