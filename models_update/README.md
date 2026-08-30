# models_update/ — trained model weights (NOT committed)

The `.pkl` weights are large and are **regenerated locally**; only this folder
structure is committed (see `.gitignore`). Files are named
`{model}_{metric}_{resolution}_{vendor}.pkl`.

| Folder            | Produced by                          | What it holds                                  |
|-------------------|--------------------------------------|------------------------------------------------|
| `if/`             | `training/training_anomaly_models.py`| Isolation Forest anomaly models                |
| `svm/`            | `training/training_anomaly_models.py`| One-Class SVM anomaly models                   |
| `lgbm/`           | `training/training_forecast_models.py`| LightGBM forecast models                      |
| `xgboost/`        | `training/training_forecast_models.py`| XGBoost forecast (baseline)                   |
| `svr/`            | `training/training_forecast_models.py`| SVR forecast models                            |
| `xgboost_stream/` | live streaming pipeline (`anomaly/streaming/`) | XGBoost incrementally updated from the stream |

Regenerate with, e.g.:
```bash
python training/training_anomaly_models.py
python training/training_forecast_models.py --models lgbm,xgb,svr
```
