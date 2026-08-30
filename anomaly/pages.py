import csv, io
from datetime import datetime
from datetime import timedelta
from django.db.models import Avg
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.views import View
from django.contrib.auth import get_user_model, update_session_auth_hash
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import Paginator
from django.db.models import F
from django.utils import timezone

from .models import Trip2017, ManualEntry

User = get_user_model()

RAW_COLUMNS = [
    ("tpep_pickup_datetime", "Pickup Datetime"),
    ("vendorid", "Vendor"),
    ("passenger_count", "Passengers"),
    ("trip_distance", "Distance"),
    ("fare_amount", "Fare ($)"),
    ("tip_amount", "Tip ($)"),
    ("tolls_amount", "Tolls ($)"),
    ("total_amount", "Total ($)"),
    ("trip_duration_minutes", "Duration (min)"),
    ("trip_speed_mph", "Speed (mph)"),
    ("price_per_distance", "Price/Dist ($/m)"),
    ("hourly_trip_volume", "Hourly Vol"),
    ("hourly_avg_speed", "Hourly Avg Speed"),
]
DB_COLUMNS = [col[0] for col in RAW_COLUMNS]

ENTRY_FIELDS = [
    ("tpep_pickup_datetime",   "Pickup datetime",  "datetime-local"),
    ("vendorid",               "Vendor",           "text"),
    ("passenger_count",        "Passengers",       "number"),
    ("trip_distance",          "Trip distance",    "number"),
    ("fare_amount",            "Fare amount",      "number"),
    ("tip_amount",             "Tip amount",       "number"),
    ("tolls_amount",           "Tolls amount",     "number"),
    ("total_amount",           "Total amount",     "number"),
    ("trip_duration_minutes",  "Duration (min)",   "number"),
    ("trip_speed_mph",         "Speed (mph)",      "number"),
    ("price_per_distance",     "Price/Dist",       "number"),
    ("hourly_trip_volume",     "Hourly volume",    "number"),
    ("hourly_avg_speed",       "Hourly avg speed", "number"),
]

_INT_FIELDS   = {"passenger_count", "hourly_trip_volume"}
_FLOAT_FIELDS = {"trip_distance", "fare_amount", "tip_amount", "tolls_amount",
                 "total_amount", "trip_duration_minutes", "trip_speed_mph", 
                 "price_per_distance", "hourly_avg_speed"}

PAGE_SIZES = [25, 50, 100]

def _parse_dt(s, default=None):
    if not s:
        return default
    s = str(s).replace("T", " ")
    try:
        return datetime.strptime(s[:16], "%Y-%m-%d %H:%M")
    except ValueError:
        return default


def _to_float(v):
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _to_int(v):
    try:
        return int(float(v)) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ── 1. Raw-data table (filters + pagination + page size + CSV export) ────────

class RawDataView(LoginRequiredMixin, View):
    template_name = "anomaly/raw_data.html"

    def _per_page(self, request):
        try:
            pp = int(request.GET.get("per_page", 50))
        except (TypeError, ValueError):
            pp = 50
        return pp if pp in PAGE_SIZES else 50

    def _collect(self, request):
        """Builds the fully-filtered row list + the display flags (shared by HTML + CSV)."""
        start = request.GET.get("start") or "2017-01-01T00:00"
        end   = request.GET.get("end")   or "2017-01-01T23:59"
        start_dt = _parse_dt(start, datetime(2017, 1, 1, 0, 0))
        end_dt   = _parse_dt(end,   datetime(2017, 1, 1, 23, 59))

        has_get = "start" in request.GET
        if not has_get:
            active_cols   = DB_COLUMNS.copy()
            show_source   = True
            show_added_by = True
        else:
            active_cols   = request.GET.getlist("cols")
            show_source   = bool(request.GET.get("col_source"))
            show_added_by = bool(request.GET.get("col_added_by"))

        vendor_filter   = request.GET.get("vendor_filter", "all")
        source_filter   = request.GET.get("source_filter", "all")
        added_by_filter = request.GET.get("added_by_filter", "all")

        # min/max only for ticked columns
        min_vals, max_vals = {}, {}
        for col_key in DB_COLUMNS:
            if col_key not in active_cols or col_key in ("tpep_pickup_datetime", "vendorid"):
                continue
            mn = request.GET.get(f"min_{col_key}")
            mx = request.GET.get(f"max_{col_key}")
            if mn:
                min_vals[col_key] = mn
            if mx:
                max_vals[col_key] = mx

        raw_list = []
        for r in (Trip2017.objects
                  .filter(tpep_pickup_datetime__range=(start_dt, end_dt))
                  .values(*DB_COLUMNS)):
            r["source"] = "raw"; r["added_by_name"] = "—"; r["added_by_id"] = None
            raw_list.append(r)

        manual_list = []
        for m in (ManualEntry.objects
                  .filter(is_deleted=False, tpep_pickup_datetime__range=(start_dt, end_dt))
                  .annotate(added_by_name=F("added_by__username"))
                  .values(*DB_COLUMNS, "added_by_name", "added_by_id")):
            m["source"] = "manual"
            manual_list.append(m)

        filtered = []
        for row in raw_list + manual_list:
            if "vendorid" in active_cols and vendor_filter != "all":
                if str(row.get("vendorid")) != str(vendor_filter):
                    continue
            if show_source and source_filter != "all":
                if row.get("source") != source_filter:
                    continue
            if show_added_by and added_by_filter != "all":
                if row.get("source") != "manual" or str(row.get("added_by_id")) != str(added_by_filter):
                    continue
            ok = True
            for col_key, mn in min_vals.items():
                v = _to_float(row.get(col_key)); fmn = _to_float(mn)
                if v is not None and fmn is not None and v < fmn:
                    ok = False; break
            if ok:
                for col_key, mx in max_vals.items():
                    v = _to_float(row.get(col_key)); fmx = _to_float(mx)
                    if v is not None and fmx is not None and v > fmx:
                        ok = False; break
            if ok:
                filtered.append(row)

        filtered.sort(key=lambda x: x["tpep_pickup_datetime"], reverse=True)
        visible_columns = [col for col in RAW_COLUMNS if col[0] in active_cols]

        return {
            "filtered": filtered, "visible_columns": visible_columns,
            "active_cols": active_cols, "show_source": show_source,
            "show_added_by": show_added_by, "min_vals": min_vals, "max_vals": max_vals,
            "vendor_filter": vendor_filter, "source_filter": source_filter,
            "added_by_filter": added_by_filter, "start": start, "end": end,
        }

    def get(self, request):
        data = self._collect(request)

        # CSV export → the WHOLE filtered set (not just the current page)
        if request.GET.get("export") == "csv":
            resp = HttpResponse(content_type="text/csv")
            resp["Content-Disposition"] = 'attachment; filename="raw_data.csv"'
            w = csv.writer(resp)
            header = [label for _, label in data["visible_columns"]]
            if data["show_source"]:   header.append("source")
            if data["show_added_by"]: header.append("added by")
            w.writerow(header)
            for row in data["filtered"]:
                line = []
                for key, _ in data["visible_columns"]:
                    val = row.get(key)
                    if hasattr(val, "strftime"):
                        val = val.strftime("%Y-%m-%d %H:%M")
                    line.append("" if val is None else val)
                if data["show_source"]:   line.append(row.get("source"))
                if data["show_added_by"]: line.append(row.get("added_by_name"))
                w.writerow(line)
            return resp

        per_page = self._per_page(request)
        page = Paginator(data["filtered"], per_page).get_page(request.GET.get("page"))

        rows = []
        for i, row in enumerate(page.object_list):
            cells = []
            for c, _ in data["visible_columns"]:
                val = row.get(c)
                if val is None:
                    cells.append("")
                elif c in _FLOAT_FIELDS:
                    try:
                        cells.append(round(float(val), 2))
                    except (ValueError, TypeError):
                        cells.append(val)
                elif hasattr(val, "strftime"):
                    cells.append(val.strftime("%Y-%m-%d %H:%M"))
                else:
                    cells.append(val)
            rows.append({"index": page.start_index() + i, "cells": cells,
                         "source": row.get("source"), "added_by": row.get("added_by_name")})

        qs = request.GET.copy(); qs.pop("page", None)
        querystring = qs.urlencode()

        return render(request, self.template_name, {
            "active": "raw",
            "raw_columns_full": RAW_COLUMNS,
            "visible_columns": data["visible_columns"],
            "show_source": data["show_source"], "show_added_by": data["show_added_by"],
            "active_cols": data["active_cols"],
            "min_vals": data["min_vals"], "max_vals": data["max_vals"],
            "vendor_filter": data["vendor_filter"], "source_filter": data["source_filter"],
            "added_by_filter": data["added_by_filter"],
            "all_users": User.objects.all().order_by("username"),
            "page": page, "rows": rows, "querystring": querystring,
            "per_page": per_page, "page_sizes": PAGE_SIZES,
            "start": data["start"], "end": data["end"],
        })


# ── 2. Data entry (add / edit / soft-delete / CSV import) ────────────────────

class DataEntryView(LoginRequiredMixin, View):
    template_name = "anomaly/data_entry.html"

    def _auto_calculate(self, obj, pickup):
        """Engine that computes on the fly the harder metrics the user did not enter"""
        if obj.trip_distance and obj.trip_duration_minutes and obj.trip_duration_minutes > 0:
            obj.trip_speed_mph = obj.trip_distance / (obj.trip_duration_minutes / 60.0)
        else:
            obj.trip_speed_mph = None

        if obj.total_amount and obj.trip_distance and obj.trip_distance > 0:
            obj.price_per_distance = obj.total_amount / obj.trip_distance
        else:
            obj.price_per_distance = None

        if pickup:
            h_start = pickup.replace(minute=0, second=0, microsecond=0)
            h_end = h_start + timedelta(hours=1)
            
            c1 = Trip2017.objects.filter(tpep_pickup_datetime__gte=h_start, tpep_pickup_datetime__lt=h_end).count()
            c2 = ManualEntry.objects.filter(is_deleted=False, tpep_pickup_datetime__gte=h_start, tpep_pickup_datetime__lt=h_end).count()
            
            obj.hourly_trip_volume = c1 + c2 + (1 if not obj.id else 0)

            agg = Trip2017.objects.filter(tpep_pickup_datetime__gte=h_start, tpep_pickup_datetime__lt=h_end).aggregate(val=Avg('trip_speed_mph'))
            obj.hourly_avg_speed = agg['val'] if agg['val'] else obj.trip_speed_mph

    def get(self, request):
        entries_qs = ManualEntry.objects.filter(is_deleted=False).select_related("added_by")
        
        member = request.GET.get("member") or ""
        if member:
            entries_qs = entries_qs.filter(added_by_id=member)

        members = (User.objects.filter(added_entries__isnull=False)
                   .distinct().order_by("username"))

        # NEW PART: format the cells in Python to avoid repeating code in the HTML
        entries = []
        for e in entries_qs.order_by("-added_at")[:500]:
            e.cells = [
                e.id,
                e.tpep_pickup_datetime.strftime("%Y-%m-%d %H:%M") if e.tpep_pickup_datetime else "",
                e.vendorid,
                e.passenger_count,
                f"{e.trip_distance:.2f}" if e.trip_distance is not None else "",
                f"{e.fare_amount:.2f}" if e.fare_amount is not None else "",
                f"{e.total_amount:.2f}" if e.total_amount is not None else "",
                f"{e.trip_duration_minutes:.1f}" if e.trip_duration_minutes is not None else "",
                f"{e.trip_speed_mph:.1f}" if e.trip_speed_mph is not None else "",
                f"{e.price_per_distance:.2f}" if e.price_per_distance is not None else "",
                e.hourly_trip_volume,
                f"{e.hourly_avg_speed:.1f}" if e.hourly_avg_speed is not None else "",
                e.added_by.username if e.added_by else "—"
            ]
            entries.append(e)

        return render(request, self.template_name, {
            "active": "entry",
            "fields": ENTRY_FIELDS,
            "entries": entries,
            "members": members,
            "member": member,
            "columns": [f[0] for f in ENTRY_FIELDS],
            "imported": request.GET.get("imported"),
        })

    def post(self, request):
        p = request.POST
        action = p.get("action")

        if action == "add":
            pickup = _parse_dt(p.get("tpep_pickup_datetime"))
            if pickup is None:
                return redirect("data_entry")
            
            obj = ManualEntry(tpep_pickup_datetime=pickup,
                              vendorid=(p.get("vendorid") or None),
                              added_by=request.user)
            
            for name in _INT_FIELDS:
                if name in p: setattr(obj, name, _to_int(p.get(name)))
            for name in _FLOAT_FIELDS:
                if name in p: setattr(obj, name, _to_float(p.get(name)))
                
            self._auto_calculate(obj, pickup)
            obj.save()

        elif action == "import":
            f = request.FILES.get("csv_file")
            n = 0
            if f:
                text = f.read().decode("utf-8-sig", errors="ignore")
                reader = csv.DictReader(io.StringIO(text))
                bulk = []
                for r in reader:
                    pickup = _parse_dt(r.get("tpep_pickup_datetime"))
                    if pickup is None:
                        continue
                    obj = ManualEntry(tpep_pickup_datetime=pickup,
                                      vendorid=(r.get("vendorid") or None),
                                      added_by=request.user)
                    for name in _INT_FIELDS:
                        if name in r: setattr(obj, name, _to_int(r.get(name)))
                    for name in _FLOAT_FIELDS:
                        if name in r: setattr(obj, name, _to_float(r.get(name)))
                    
                    self._auto_calculate(obj, pickup)
                    bulk.append(obj)
                    n += 1
                if bulk:
                    ManualEntry.objects.bulk_create(bulk, batch_size=1000)
            return redirect(f"/data/entry/?imported={n}")

        elif action == "edit":
            entry = ManualEntry.objects.filter(id=p.get("entry_id"), is_deleted=False).first()
            if entry and (request.user.is_superuser or entry.added_by == request.user):
                pickup = _parse_dt(p.get("tpep_pickup_datetime"))
                if pickup:
                    entry.tpep_pickup_datetime = pickup
                entry.vendorid = p.get("vendorid") or None
                for name in _INT_FIELDS:
                    if name in p: setattr(entry, name, _to_int(p.get(name)))
                for name in _FLOAT_FIELDS:
                    if name in p: setattr(entry, name, _to_float(p.get(name)))
                
                self._auto_calculate(entry, pickup or entry.tpep_pickup_datetime)
                entry.save()

        elif action == "delete":
            entry = ManualEntry.objects.filter(id=p.get("id"), is_deleted=False).first()
            if entry and (request.user.is_superuser or entry.added_by == request.user):
                entry.is_deleted = True
                entry.deleted_by = request.user
                entry.deleted_at = timezone.now()
                entry.save(update_fields=["is_deleted", "deleted_by", "deleted_at"])

        return redirect("data_entry")

# ── 3. Activity / audit log ──────────────────────────────────────────────────

class ActivityLogView(LoginRequiredMixin, View):
    template_name = "anomaly/activity_log.html"
    PER_PAGE = 50

    @staticmethod
    def _summary(e):
        return (f"#{e.id} · {e.tpep_pickup_datetime:%Y-%m-%d %H:%M} · "
                f"vendor {e.vendorid or '—'} · fare {e.fare_amount if e.fare_amount is not None else '—'}")

    def get(self, request):
        member = request.GET.get("member") or ""
        action = request.GET.get("action") or "all"   # all | added | deleted

        events = []
        for e in ManualEntry.objects.select_related("added_by", "deleted_by"):
            if e.added_at and action in ("all", "added"):
                events.append({"time": e.added_at, "action": "added",
                               "user": e.added_by.username if e.added_by else "—",
                               "user_id": e.added_by_id, "summary": self._summary(e)})
            if e.is_deleted and e.deleted_at and action in ("all", "deleted"):
                events.append({"time": e.deleted_at, "action": "deleted",
                               "user": e.deleted_by.username if e.deleted_by else "—",
                               "user_id": e.deleted_by_id, "summary": self._summary(e)})

        if member:
            events = [ev for ev in events if str(ev["user_id"]) == str(member)]
        events.sort(key=lambda x: x["time"], reverse=True)

        page = Paginator(events, self.PER_PAGE).get_page(request.GET.get("page"))
        members = User.objects.all().order_by("username")

        qs = request.GET.copy(); qs.pop("page", None)

        return render(request, self.template_name, {
            "active": "activity",
            "page": page, "members": members,
            "member": member, "action_filter": action,
            "querystring": qs.urlencode(),
        })


# ── 4. Profile ───────────────────────────────────────────────────────────────

class ProfileView(LoginRequiredMixin, View):
    template_name = "anomaly/profile.html"

    def _summary(self, user):
        added_all = ManualEntry.objects.filter(added_by=user)
        last = added_all.order_by("-added_at").first()
        return {
            "added_total":  added_all.count(),
            "added_active": added_all.filter(is_deleted=False).count(),
            "deleted_by_me": ManualEntry.objects.filter(deleted_by=user).count(),
            "last_added": last.added_at if last else None,
        }

    def get(self, request):
        return render(request, self.template_name, {
            "active": "profile",
            "form": PasswordChangeForm(user=request.user),
            "summary": self._summary(request.user),
            "saved": request.GET.get("saved") == "1",
        })

    def post(self, request):
        form = PasswordChangeForm(user=request.user, data=request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            return redirect("/profile/?saved=1")
        return render(request, self.template_name, {
            "active": "profile", "form": form,
            "summary": self._summary(request.user), "saved": False,
        })