from django.db import models
from django.conf import settings


# NOTE: The two trip tables keep their column names as-is. These names are the
# "metric vocabulary" shared with the trained .pkl files (e.g. lgbm_fare_amount_...,
# lgbm_sure_dk_...). Renaming them would break the trained models, so field names
# mirror the real database columns here.


class TrainTrip(models.Model):                       
    vendorid              = models.TextField()       
    tpep_pickup_datetime  = models.DateTimeField(db_index=True)
    tpep_dropoff_datetime = models.DateTimeField()
    passenger_count       = models.BigIntegerField(null=True)
    trip_distance         = models.FloatField(null=True)
    fare_amount           = models.FloatField(null=True)
    tip_amount            = models.FloatField(null=True)
    tolls_amount          = models.FloatField(null=True)
    total_amount          = models.FloatField(null=True)
    pulocationid          = models.BigIntegerField(null=True)
    dolocationid          = models.BigIntegerField(null=True)
    
    trip_duration_minutes = models.FloatField(null=True)   
    trip_speed_mph        = models.FloatField(null=True)   
    price_per_distance    = models.FloatField(null=True)
    hourly_trip_volume    = models.IntegerField(null=True)
    hourly_avg_speed      = models.FloatField(null=True)

    class Meta:
        managed  = True  
        db_table = "sefer_egitim"

    @property
    def pickup_day_name(self):
      
        if self.tpep_pickup_datetime:
            return self.tpep_pickup_datetime.strftime('%A')
        return None


class Trip2017(models.Model):                        # 2017 data (panel source)
    vendorid              = models.TextField()       
    tpep_pickup_datetime  = models.DateTimeField(db_index=True)
    tpep_dropoff_datetime = models.DateTimeField()
    passenger_count       = models.BigIntegerField(null=True)
    trip_distance         = models.FloatField(null=True)
    fare_amount           = models.FloatField(null=True)
    tip_amount            = models.FloatField(null=True)
    tolls_amount          = models.FloatField(null=True)
    total_amount          = models.FloatField(null=True)
    pulocationid          = models.BigIntegerField(null=True)
    dolocationid          = models.BigIntegerField(null=True)

    trip_duration_minutes = models.FloatField(null=True)   
    trip_speed_mph        = models.FloatField(null=True)   
    price_per_distance    = models.FloatField(null=True)
    hourly_trip_volume    = models.IntegerField(null=True)
    hourly_avg_speed      = models.FloatField(null=True)
    
    class Meta:
        managed  = False
        db_table = "sefer_2017"

    @property
    def pickup_day_name(self):
        if self.tpep_pickup_datetime:
            return self.tpep_pickup_datetime.strftime('%A')
        return None


# For the tables below, English field names are mapped onto the existing
# (Turkish) database columns via db_column — no DB change needed.

class TrainingProfile(models.Model):                
    metric     = models.TextField(db_column="metrik", primary_key=True)
    resolution = models.TextField(db_column="coz")
    vendor     = models.TextField(db_column="vendor")
    wd         = models.SmallIntegerField(db_column="wd")
    tb         = models.IntegerField(db_column="tb")          # seconds since midnight
    expected   = models.FloatField(db_column="bek")
    mad        = models.FloatField(db_column="mad")

    class Meta:
        managed  = False
        db_table = "egitim_profil"


class AnomalyResult(models.Model):                   # anomali_sonuc (cache)
    time         = models.DateTimeField(db_column="zaman", db_index=True)
    metric       = models.TextField(db_column="metrik")
    resolution   = models.TextField(db_column="coz")
    vendor       = models.TextField(db_column="vendor")
    model_name   = models.TextField(db_column="model_adi")
    z_score      = models.FloatField(db_column="z_skoru", null=True)
    actual_value = models.FloatField(db_column="gercek_deger")
    expected     = models.FloatField(db_column="beklenen", null=True)
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed  = False           # table already exists; excluded from migrations
        db_table = "anomali_sonuc"


class ForecastResult(models.Model):                  # forecast result cache (DB table: tahmin_sonuc)
    time         = models.DateTimeField(db_column="zaman", db_index=True)
    metric       = models.TextField(db_column="metrik")
    resolution   = models.TextField(db_column="coz")
    vendor       = models.TextField(db_column="vendor")
    model_name   = models.TextField(db_column="model_adi")
    forecast     = models.FloatField(db_column="tahmin")
    actual_value = models.FloatField(db_column="gercek", null=True)
    horizon      = models.IntegerField()
    created_at   = models.DateTimeField(auto_now_add=True)

    class Meta:
        managed  = False           # table already exists; excluded from migrations
        db_table = "tahmin_sonuc"


# ── Manually entered trip records (2018+) with an audit trail ────────────────
# Same data columns as Trip2017 so the raw-data page can merge both cleanly.
# Django-managed (needs a migration to create the table). Removal is a SOFT
# delete: the row stays, is_deleted is flipped and deleted_by/deleted_at record
# who removed it and when — so "who added / who removed" is always auditable.

class ManualEntry(models.Model):

    vendorid              = models.TextField(null=True, blank=True)
    tpep_pickup_datetime  = models.DateTimeField()
    passenger_count       = models.BigIntegerField(null=True)
    trip_distance         = models.FloatField(null=True)
    fare_amount           = models.FloatField(null=True)
    tip_amount            = models.FloatField(null=True)
    tolls_amount          = models.FloatField(null=True)
    total_amount          = models.FloatField(null=True)
    pulocationid          = models.BigIntegerField(null=True)
    dolocationid          = models.BigIntegerField(null=True)
    
    trip_duration_minutes = models.FloatField(null=True)   
    trip_speed_mph        = models.FloatField(null=True)   
    price_per_distance    = models.FloatField(null=True)
    hourly_trip_volume    = models.IntegerField(null=True)
    hourly_avg_speed      = models.FloatField(null=True)

    # audit trail
    added_by   = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                   null=True, related_name="added_entries")
    added_at   = models.DateTimeField(auto_now_add=True)
    is_deleted = models.BooleanField(default=False)
    deleted_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
                                   null=True, blank=True, related_name="deleted_entries")
    deleted_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "manual_entry"
        ordering = ["-added_at"]