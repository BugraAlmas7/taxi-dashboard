from datetime import datetime

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views import View
from django.contrib.auth.mixins import LoginRequiredMixin
import pandas as pd
import plotly.graph_objects as go

from .models import Trip2017
from . import time_series as ts
from .anomaliy_calculation import MODELS as ANOMALY_MODELS
from .forecast_calculation import MODELS as FORECAST_MODELS

from django.contrib.auth import login as auth_login
from django.contrib.auth.forms import UserCreationForm


_CFG = {"scrollZoom": True}

# ── Dynamic lists (from the backend instead of static HTML) ──────────────────

# Numeric fields NOT used as a metric
# Numeric fields NOT used as a metric (legacy columns added here too!)
_METRIC_EXCLUDE = {
    "id", "vendorid", "pulocationid", "dolocationid",
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "sure_dk", "hiz_mph" # <-- added these
}
_NUMERIC_TYPES = {
    "FloatField", "IntegerField", "BigIntegerField", "SmallIntegerField",
    "DecimalField", "PositiveIntegerField", "PositiveBigIntegerField",
}

# Resolution / mode / calibration lists: (value, label)
RESOLUTION_LIST = [
    ("1s", "1 sec"), ("3s", "3 sec"), ("5s", "5 sec"), ("10s", "10 sec"),
    ("15s", "15 sec"), ("30s", "30 sec"),
    ("1m", "1 min"), ("3m", "3 min"), ("5m", "5 min"), ("10m", "10 min"),
    ("15m", "15 min"), ("30m", "30 min"),
    ("1h", "1 hour"), ("3h", "3 hours"), ("1d", "Daily"),
]
MODE_LIST = [
    ("chart", "Chart"), ("anomaly", "Anomaly"),
    ("forecast", "Forecast"), 
]
# NOTE: value "yok" (not "none") — calibration.py's LEVELS map uses "yok" for
# "no band". With "none" the band never turned off (valid_level fell back to 95%).
CALIBRATION_LIST = [("95", "95% band"), ("90", "90% band"), ("yok", "None")]

ANOMALY_LABELS = {
    "mad": "MAD (global)", "mad_lokal": "MAD (local)",
    "if": "Isolation Forest", "svm": "One-Class SVM",
    "dbscan": "DBSCAN", "lstm": "LSTM",
}
FORECAST_LABELS = {
    "lgbm": "LightGBM", "timesfm": "TimesFM", "chronos": "Chronos",
    "xgboost": "XGBoost", "svr": "SVR",
    "timesfm_ft": "TimesFM (FT)", "chronos_ft": "Chronos-2 (FT)",
    "timesfm_ft_optuna": "TimesFM (Optuna)",
    "chronos_ft_optuna": "Chronos-2 (Optuna)",
    # NOTE: the "live" stream models (xgboost_stream / chronos_stream) live on the
    # dedicated /live/ page now, not in this dropdown.
}


# ── Helper functions (not views; the classes below use them) ─────────────────

def _metric_list():
    """Metrics derived from the model's numeric fields (+ synthetic 'sefer')."""
    fields = [
        f.name for f in Trip2017._meta.fields
        if f.get_internal_type() in _NUMERIC_TYPES and f.name not in _METRIC_EXCLUDE
    ]
    return ["sefer"] + fields


def _vendor_list():
    """Vendor options (kept static for speed).
    A DISTINCT scan over the ~millions-row trip_2017 table (unindexed text
    column) freezes the page on every load, so we list the known vendors."""
    return [("hepsi", "All"), ("1", "Vendor 1"), ("2", "Vendor 2")]


def _resolution_to_period(resolution):
    mapping = {
        "1m": "minute", "3m": "minute", "5m": "minute",
        "10m": "minute", "15m": "minute", "30m": "minute",
        "1h": "hour", "3h": "hour", "1d": "day",
    }
    return mapping.get(resolution, "hour")


def _dt(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def _trip_qs(vendor, start, end):
    """trip_2017 + [start,end] + vendor filter (parameterized, no SQL)."""
    qs = Trip2017.objects.filter(tpep_pickup_datetime__range=(_dt(start), _dt(end)))
    if vendor != "hepsi":
        qs = qs.filter(vendorid=vendor)
    return qs


def _raw_rows(vendor, start, end, metric):
    columns = ["tpep_pickup_datetime"] + ([] if metric == "sefer" else [metric])
    return _trip_qs(vendor, start, end).values(*columns)


def _compute(mode, anomaly_model, forecast_model, calibration, metric, resolution, vendor,
             start, end, _vf, period, k, start_date, start_time, end_date, end_time):
    args = (metric, resolution, vendor, start, end, _vf, period, k,
            start_date, start_time, end_date, end_time)

    if mode == "chart":
        p = ts.bin_rows(_raw_rows(vendor, start, end, metric), metric,
                        ts.resolution_freq(resolution), how="median")
        if len(p) == 0:
            return "<p style='color:#c0392b;padding:12px'>No data in this range</p>"
        fig = go.Figure(go.Scatter(
            x=p["zaman"], y=p["deger"], mode="lines+markers",
            line=dict(color="#1C7293", width=1.5), marker=dict(size=4),
            hovertemplate="%{x}<br><b>%{y:.1f}</b><extra></extra>"
        ))
        fig.update_layout(
            title=f"{metric} . {resolution} . {start_date} {start_time}->{end_date} {end_time} . vendor:{vendor}",
            height=460, template="plotly_white"
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config=_CFG)

    elif mode == "anomaly":
        fn = ANOMALY_MODELS.get(anomaly_model)
        if fn:
            return fn(*args)
        return f"<p style='color:#c0392b;padding:12px'>Unknown model: {anomaly_model}</p>"

    elif mode == "forecast":
        fn = FORECAST_MODELS.get(forecast_model)
        if fn:
            return fn(*args, calibration)
        return f"<p style='color:#c0392b;padding:12px'>Unknown model: {forecast_model}</p>"

    elif mode == "raw":
        if metric == "sefer":
            return "<p style='color:#c0392b;padding:12px'>'sefer' is meaningless at record level.</p>"
        rows = (
            _trip_qs(vendor, start, end)
            .values("tpep_pickup_datetime", "trip_distance", "trip_duration_minutes", metric)[:30001]
        )
        d = pd.DataFrame(list(rows))
        if len(d) == 0:
            return "<p style='color:#c0392b;padding:12px'>No records in this range</p>"
        d = d.dropna(subset=[metric])
        if len(d) == 0:
            return "<p style='color:#c0392b;padding:12px'>No records in this range</p>"
        if len(d) > 30000:
            return f"<p style='color:#c0392b;padding:12px'>{len(d):,} records — narrow the window.</p>"
        v = d[metric].astype(float)
        med = v.median()
        mad = (v - med).abs().median() or 1e-9
        d["z"] = 0.6745 * (v - med) / mad
        d["anom"] = d["z"].abs() > k
        normal = d[~d["anom"]]
        outlier = d[d["anom"]]
        fig = go.Figure()
        fig.add_scatter(x=normal["tpep_pickup_datetime"], y=normal[metric], mode="markers",
                        name="normal", marker=dict(color="#1C7293", size=5, opacity=0.6))
        fig.add_scatter(x=outlier["tpep_pickup_datetime"], y=outlier[metric], mode="markers",
                        name=f"outlier({len(outlier)})", marker=dict(color="orange", size=8))
        fig.update_layout(title=f"{metric} . record . {start_date} {start_time}->{end_date} {end_time} . k={k}",
                          height=460, template="plotly_white")
        return fig.to_html(full_html=False, include_plotlyjs="cdn", config=_CFG)

    return ""


def _panel_context(selected=None):
    """Builds all dynamic lists + selected values for the panel."""
    selected = selected or {}
    ctx = {
        "metric_list":      _metric_list(),
        "vendor_list":      _vendor_list(),
        "resolution_list":  RESOLUTION_LIST,
        "mode_list":        MODE_LIST,
        "calibration_list": CALIBRATION_LIST,
        "anomaly_list":     [(k, ANOMALY_LABELS.get(k, k)) for k in ANOMALY_MODELS],
        "forecast_list":    [(k, FORECAST_LABELS.get(k, k)) for k in FORECAST_MODELS],
        "panels": [1, 2, 3, 4, 5],
        # selected / default values
        "metric": "sefer", "start_date": "2017-01-01", "end_date": "2017-01-07",
        "start_time": "00:00", "end_time": "23:59", "resolution": "1m",
        "vendor": "hepsi", "mode": "chart", "anomaly_model": "mad",
        "forecast_model": "lgbm", "calibration": "95", "k": 3.5,
        "chart": None,
    }
    ctx.update(selected)
    return ctx


# ── Class-based views ─────────────────────────────────────────────────────────

class RegisterView(View):
    """Sign-up page (class-based). GET → empty form, POST → save + auto-login."""
    template_name = "anomaly/register.html"

    def get(self, request):
        return render(request, self.template_name, {"form": UserCreationForm()})

    def post(self, request):
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            auth_login(request, user)          # auto-login after sign-up
            return redirect("panel")
        return render(request, self.template_name, {"form": form})


class PanelView(LoginRequiredMixin, View):
    """
    Main panel (class-based). Login required (LoginRequiredMixin → settings.LOGIN_URL).
      GET  → empty panel
      POST → build the chart for the chosen mode; JSON if _ajax=1, else full page.
    """
    template_name = "anomaly/panel.html"

    def get(self, request):
        return render(request, self.template_name, _panel_context())

    def post(self, request):
        p = request.POST
        metric         = p.get("metric",         "sefer")
        start_date     = p.get("start_date",     "2017-01-01")
        start_time     = p.get("start_time",     "00:00")
        end_date       = p.get("end_date",       "2017-01-07")
        end_time       = p.get("end_time",       "23:59")
        resolution     = p.get("resolution",     "1h")
        vendor         = p.get("vendor",         "hepsi")
        mode           = p.get("mode",           "chart")
        anomaly_model  = p.get("anomaly_model",  "mad")
        forecast_model = p.get("forecast_model", "lgbm")
        calibration    = p.get("calibration",    "95")
        # k box may be empty (e.g. not used in Forecast mode) → fall back to default
        try:
            k = float(p.get("k") or 3.5)
        except (TypeError, ValueError):
            k = 3.5

        start = f"{start_date} {start_time}"
        end   = f"{end_date} {end_time}"
        _vf   = ""   # unused; kept for signature compatibility
        period = _resolution_to_period(resolution)
        vendor = "hepsi" if vendor == "hepsi" else vendor

        chart = _compute(mode, anomaly_model, forecast_model, calibration, metric,
                         resolution, vendor, start, end, _vf, period, k,
                         start_date, start_time, end_date, end_time)

        if p.get("_ajax") == "1":
            return JsonResponse({"html": chart or ""})

        context = _panel_context({
            "chart": chart,
            "metric": metric, "start_date": start_date, "end_date": end_date,
            "start_time": start_time, "end_time": end_time,
            "resolution": resolution, "vendor": vendor, "mode": mode,
            "anomaly_model": anomaly_model, "forecast_model": forecast_model,
            "calibration": calibration, "k": k,
        })
        return render(request, self.template_name, context)