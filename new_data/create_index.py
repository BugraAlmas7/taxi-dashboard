import os
import django

# Set up the Django environment
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
django.setup()

from django.db import connection
from django.db.models import Index
from anomaly.models import TrainTrip

def run_indexing():
    print("Starting the indexing process via pure Django ORM (Schema Editor)...")
    print("Please wait. Depending on your SSD speed, this operation may take 10-20 minutes.\n")

    # Define the indexes purely as Django Python objects
    time_index = Index(fields=['tpep_pickup_datetime'], name='idx_sefer_zaman')
    vendor_index = Index(fields=['vendorid'], name='idx_sefer_vendor')

    # Apply the indexes directly to the database schema using Django's SchemaEditor
    with connection.schema_editor() as schema_editor:
        print("1/2: Adding time index to the TrainTrip model...")
        schema_editor.add_index(TrainTrip, time_index)
        print("Time index added successfully!\n")

        print("2/2: Adding vendor index to the TrainTrip model...")
        schema_editor.add_index(TrainTrip, vendor_index)
        print("Vendor index added successfully!\n")

    print("All indexing operations finished successfully!")
    print("Zero raw SQL was used. The database is now production-ready.")

if __name__ == "__main__":
    run_indexing()