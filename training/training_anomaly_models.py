"""
training_anomaly_models.py
- Includes all current metrics.
- Adjusted to the models_update folder layout.
- Each resolution (1m, 3m, 5m, ...) is processed separately.
- Fixed the sure_dk / hiz_mph database errors.
"""
import os, sys, pickle, gc

# ── Bootstrap Django ORM ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
import django
django.setup()

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from django.db.models import Count, Avg
from django.db.models.functions import TruncMinute, TruncHour, TruncDate

from anomaly.models import TrainTrip, TrainingProfile

# Model output folder layout
MODEL_DIR = os.path.join(PROJECT_DIR, "models_update")
print(f"Model dir: {MODEL_DIR}\n")

# Metrics (legacy columns removed)
METRICS = [
    "sefer", "passenger_count", "trip_distance", "fare_amount",
    "tip_amount", "tolls_amount", "total_amount", 
    "trip_duration_minutes", "trip_speed_mph", "price_per_distance",
    "hourly_trip_volume", "hourly_avg_speed"
]
VENDOR_LIST = ["hepsi", "1", "2"]

# Every resolution is now processed separately
RESOLUTIONS = {
    "1m": 60, "3m": 180, "5m": 300, "10m": 600, "15m": 900, "30m": 1800,
    "1h": 3600, "3h": 10800, "1d": 86400
}

IF_SAMPLE = 1_000_000
IF_CONTAM = 0.05
SVM_NU    = 0.05

print("clearing training_profile rows (ORM)...", end=" ", flush=True)
TrainingProfile.objects.all().delete()
print("done.\n")

for vendor in VENDOR_LIST:
    print(f"{'='*55}\nVENDOR: {vendor}")
    
    for resolution, dk in RESOLUTIONS.items():
        print(f"  [{vendor}] resolution={resolution} aggregating...", end=" ", flush=True)

        qs = TrainTrip.objects.all()
        if vendor != "hepsi":
            qs = qs.filter(vendorid=vendor)
        
        # Database truncation settings
        if dk < 3600:
            trunc_col = TruncMinute('tpep_pickup_datetime')
            feats = ["deger", "wd", "saat", "dakika"]
        elif dk < 86400:
            trunc_col = TruncHour('tpep_pickup_datetime')
            feats = ["deger", "wd", "saat"]
        else:
            trunc_col = TruncDate('tpep_pickup_datetime')
            feats = ["deger", "wd"]

        # Fixed version of the query that used to blow up
        qs_binned = qs.annotate(
            time_group=trunc_col
        ).values('time_group').annotate(
            sefer=Count('pk'),
            passenger_count=Avg('passenger_count'),
            trip_distance=Avg('trip_distance'),
            fare_amount=Avg('fare_amount'),
            tip_amount=Avg('tip_amount'),
            tolls_amount=Avg('tolls_amount'),
            total_amount=Avg('total_amount'),
            trip_duration_minutes=Avg('trip_duration_minutes'),
            trip_speed_mph=Avg('trip_speed_mph'),
            price_per_distance=Avg('price_per_distance'),
            hourly_trip_volume=Avg('hourly_trip_volume'),
            hourly_avg_speed=Avg('hourly_avg_speed')
        )

        binned = pd.DataFrame.from_records(qs_binned.iterator(chunk_size=50000))

        if len(binned) == 0:
            print("no data.")
            continue

        binned['time_group'] = pd.to_datetime(binned['time_group'])
        
        # Clean resampling to intermediate resolutions (3m, 5m, ...) via pandas
        if dk not in [60, 3600, 86400]:
            freq = f"{dk}s"
            binned = binned.set_index('time_group').resample(freq).mean(numeric_only=True).reset_index()

        binned['wd'] = ((binned['time_group'].dt.weekday + 1) % 7).astype('int64')

        if dk >= 86400:
            binned['tb'] = 0
        else:
            binned['tb'] = (binned['time_group'].dt.hour * 3600
                            + binned['time_group'].dt.minute * 60)
        binned['tb'] = binned['tb'].astype('int64')

        profile_objs = []
        for metric in METRICS:
            sub = binned[["wd", "tb", metric]].dropna(subset=[metric])
            if len(sub) == 0:
                continue
            expected = sub.groupby(["wd", "tb"])[metric].median().rename("expected").reset_index()
            m2 = sub.merge(expected, on=["wd", "tb"])
            m2["dev"] = (m2[metric] - m2["expected"]).abs()
            mad = m2.groupby(["wd", "tb"])["dev"].median().clip(lower=1e-9).rename("mad").reset_index()
            prof = expected.merge(mad, on=["wd", "tb"])

            for r in prof.itertuples(index=False):
                profile_objs.append(TrainingProfile(
                    metric=metric, resolution=resolution, vendor=vendor,
                    wd=int(r.wd), tb=int(r.tb),        
                    expected=float(r.expected), mad=float(r.mad),
                ))
        
        TrainingProfile.objects.bulk_create(profile_objs, batch_size=5000)
        n_prof = len(profile_objs)

        df = binned.sample(min(IF_SAMPLE, len(binned)), random_state=42).copy()
        df["saat"]   = df["tb"] // 3600
        df["dakika"] = (df["tb"] % 3600) // 60

        n_if = n_svm = 0
        for metric in METRICS:
            feat_list = [f for f in feats if f != "saniye"]
            
            df_m = df[["wd", "saat", "dakika", metric]].dropna(subset=[metric])
            if len(df_m) < 10:
                continue
            X_m = df_m.rename(columns={metric: "deger"})[feat_list].values.astype(float)
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X_m)

            if_model = IsolationForest(n_estimators=100, contamination=IF_CONTAM, random_state=42)
            if_model.fit(X_s)

            svm_model = None
            if dk >= 3600:  
                svm_model = OneClassSVM(nu=SVM_NU, kernel="rbf", gamma="scale").fit(X_s)

            if_filename = f"if_{metric}_{resolution}_{vendor}.pkl"
            if_path = os.path.join(MODEL_DIR, "if", if_filename)
            os.makedirs(os.path.dirname(if_path), exist_ok=True)
            with open(if_path, "wb") as f:
                pickle.dump({"model": if_model, "scaler": scaler, "features": feat_list}, f)
            
            if svm_model is not None:
                svm_filename = f"svm_{metric}_{resolution}_{vendor}.pkl"
                svm_path = os.path.join(MODEL_DIR, "svm", svm_filename)
                os.makedirs(os.path.dirname(svm_path), exist_ok=True)
                with open(svm_path, "wb") as f:
                    pickle.dump({"model": svm_model, "scaler": scaler, "features": feat_list}, f)
            
            n_if += 1
            if svm_model is not None:
                n_svm += 1

        print(f"prof={n_prof:,} IF({n_if}) SVM({n_svm}) saved.")
    gc.collect()

print(f"\n{'='*55}\nDONE! All models saved to: {MODEL_DIR}")