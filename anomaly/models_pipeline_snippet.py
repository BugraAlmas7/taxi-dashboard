# ─────────────────────────────────────────────────────────────────────────────
# APPEND THIS to <app>/models.py  (e.g. anomaly/models.py), then:
#     python manage.py makemigrations
#     python manage.py migrate
# The streaming orchestrator writes one row here per model per window; the
# dashboard polls it to show WAPE-over-time + anomaly counts live.
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWindowResult(models.Model):
    run_name     = models.CharField(max_length=64, db_index=True)
    window_index = models.IntegerField()
    ts_start     = models.DateTimeField(db_index=True)   # window's first trip time
    ts_end       = models.DateTimeField()                # window's last trip time

    metric       = models.TextField()
    resolution   = models.TextField()
    vendor       = models.TextField()

    # anomaly cleaning stats (same for every model row of a window)
    n_raw        = models.IntegerField()
    n_anomaly    = models.IntegerField()
    n_clean      = models.IntegerField()
    anom_rate    = models.FloatField()

    # per-model update outcome
    model_name   = models.CharField(max_length=32)       # xgboost_stream / chronos_stream
    mode         = models.CharField(max_length=16)        # finetune / train
    status       = models.CharField(max_length=16)        # ok / warmup / skip / error
    wape         = models.FloatField(null=True)
    mae          = models.FloatField(null=True)
    n_train      = models.IntegerField(null=True)
    detail       = models.CharField(max_length=300, blank=True, default="")

    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "pipeline_window_result"
        ordering = ["run_name", "window_index", "model_name"]
        indexes = [models.Index(fields=["run_name", "window_index"])]
