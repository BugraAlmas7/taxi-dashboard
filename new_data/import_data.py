import os
import django
import csv
from datetime import datetime

# Bootstrap the Django environment from outside the project
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
django.setup()

from anomaly.models import TrainTrip

def run():
    csv_path = "D:/data/engineered_2015_2016.csv"
    batch_size = 10000  # send in 10k batches so we don't overload the database

    print(f"BULK IMPORT STARTING: {csv_path}")

    # Collect the model's valid column names (lowercased) into a list
    valid_fields = [f.name for f in TrainTrip._meta.fields]

    trips = []
    total_inserted = 0

    # Read the file line by line to keep RAM usage low
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Keep only the columns that exist on our model
            clean_row = {}
            for k, v in row.items():
                k_lower = k.lower()
                # Add the column only if it exists on the model and is not empty
                if k_lower in valid_fields and v.strip() != "":
                    clean_row[k_lower] = v

            trips.append(TrainTrip(**clean_row))

            # Once the batch reaches 10,000, flush it to the database (no raw SQL!)
            if len(trips) >= batch_size:
                TrainTrip.objects.bulk_create(trips, batch_size=batch_size)
                total_inserted += len(trips)
                trips = []  # reset the list (free RAM)
                print(f"{total_inserted:,} rows imported... ({datetime.now().strftime('%H:%M:%S')})")

        # Flush the final partial batch after the loop ends
        if trips:
            TrainTrip.objects.bulk_create(trips, batch_size=batch_size)
            total_inserted += len(trips)

    print(f"DONE! Import complete. {total_inserted:,} rows loaded in total.")

if __name__ == "__main__":
    run()
