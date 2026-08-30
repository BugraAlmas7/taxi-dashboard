import os
import sys
import django
import time
import csv
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
django.setup()

from django.db import connection
from anomaly.models import Trip2017


def run():
    csv_path = "D:/data/engineered_2017.csv"
    batch_size = 10000

    print(f"BULK IMPORT STARTING: {csv_path}")

    valid_fields = [f.name for f in Trip2017._meta.fields]

    trips = []
    total_inserted = 0

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            clean_row = {}
            for k, v in row.items():
                k_lower = k.lower()
                if k_lower in valid_fields and v.strip() != "":
                    clean_row[k_lower] = v

            trips.append(Trip2017(**clean_row))

            if len(trips) >= batch_size:
                Trip2017.objects.bulk_create(trips, batch_size=batch_size)
                total_inserted += len(trips)
                trips = []
                print(f"{total_inserted:,} rows imported... ({datetime.now().strftime('%H:%M:%S')})")

        if trips:
            Trip2017.objects.bulk_create(trips, batch_size=batch_size)
            total_inserted += len(trips)

    print(f"DONE! 2017 import complete. {total_inserted:,} rows loaded in total.")

if __name__ == "__main__":
    run()
