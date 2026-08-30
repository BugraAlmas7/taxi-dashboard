# notebooks/ — Colab fine-tuning notebooks

Run on Colab (GPU). All cell outputs were cleared before committing.

- `data_for_finetune.ipynb` — build `taxi_series.parquet` from the cleaned trips
- `chronos_ft.ipynb` — Chronos-2 LoRA fine-tune
- `chronos_ft_optuna.ipynb`, `ch_ft_optuna.ipynb` — Optuna HPO for Chronos-2
- `timesfm_ft.ipynb` — TimesFM-2.5 LoRA fine-tune
- `timesfm_ft_optuna.ipynb`, `tfm_ft_optuna.ipynb` — Optuna HPO for TimesFM

Set your own Hugging Face / W&B credentials at runtime — none are stored here.
