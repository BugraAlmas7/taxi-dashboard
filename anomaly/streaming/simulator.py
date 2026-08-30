"""
streaming/simulator.py
Replays a trip table as a stream: yields raw-trip batches in time order, paging
with a (time, id) cursor so it never loads the whole table into memory. This is
the "sensor" of the pipeline — swap it for a Kafka/HTTP consumer later without
touching the rest.
"""
from datetime import datetime
from django.db.models import Q
from ..models import Trip2017, TrainTrip

_SOURCES = {"trip2017": Trip2017, "train": TrainTrip}

# raw columns each batch carries (numeric features + the two timestamps)
_COLUMNS = [
    "id", "tpep_pickup_datetime", "tpep_dropoff_datetime", "vendorid",
    "trip_distance", "trip_duration_minutes", "trip_speed_mph",
    "total_amount", "fare_amount", "price_per_distance", "passenger_count",
]


class StreamSimulator:
    """Iterate raw trips oldest-first, `batch_size` rows at a time."""

    def __init__(self, cfg):
        self.model = _SOURCES.get(cfg.source, Trip2017)
        self.batch_size = cfg.batch_size
        self.vendor = cfg.vendor
        self._t = datetime.strptime(cfg.start, "%Y-%m-%d %H:%M")
        self._last_id = -1        # tie-breaker within the same timestamp

    def __iter__(self):
        return self

    def __next__(self):
        qs = self.model.objects.filter(tpep_pickup_datetime__gte=self._t)
        if self.vendor != "hepsi":
            qs = qs.filter(vendorid=self.vendor)
        # (time, id) cursor: strictly after the last row we returned
        # (time, id) cursor: strictly after the last row we returned
        qs = qs.filter(
            Q(tpep_pickup_datetime__gt=self._t) |
            Q(tpep_pickup_datetime=self._t, id__gt=self._last_id)
        )
        rows = list(qs.order_by("tpep_pickup_datetime", "id")
                      .values(*_COLUMNS)[: self.batch_size])
        if not rows:
            raise StopIteration
        last = rows[-1]
        self._t = last["tpep_pickup_datetime"]
        self._last_id = last["id"]
        return rows
