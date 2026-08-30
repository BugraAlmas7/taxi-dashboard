import os
import sys
import django
import time

# ── Bootstrap the Django ORM (trick to reach the project root) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)  # go up one level (to the taxi project root)
sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "taxi_web.settings")
django.setup()

from django.db import connection
from django.db.models import Index
from anomaly.models import Trip2017

# ... the rest of the code continues from here ...
def run_indexing():
    print("Starting the indexing process via pure Django ORM (Schema Editor)...")
    print("Please wait. Depending on your SSD speed, this operation may take a few minutes.\n")

    time_index = Index(fields=['tpep_pickup_datetime'], name='idx_sefer2017_zaman')
    vendor_index = Index(fields=['vendorid'], name='idx_sefer2017_vendor')

    with connection.schema_editor() as schema_editor:
        print("1/2: Adding time index to the Trip2017 model...")
        schema_editor.add_index(Trip2017, time_index)
        print("Time index added successfully!\n")

        print("2/2: Adding vendor index to the Trip2017 model...")
        schema_editor.add_index(Trip2017, vendor_index)
        print("Vendor index added successfully!\n")

    print("All indexing operations finished successfully!")
    print("Zero raw SQL was used. The database is now production-ready.")

if __name__ == "__main__":
    run_indexing()
