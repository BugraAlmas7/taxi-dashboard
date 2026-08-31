"""
taxi_web/urls.py
Project URL routes (wired to class-based views).
"""
from django.contrib import admin
from django.urls import path
from django.contrib.auth import views as auth_views
from anomaly import views_live
from anomaly import views as av
from anomaly import pages
from anomaly import views_pipeline


urlpatterns = [
    path("admin/", admin.site.urls),

    # Main dashboard
    path("", av.PanelView.as_view(), name="panel"),

    # Extra screens

    path("data/",       pages.RawDataView.as_view(),     name="raw_data"),      # + ?export=csv
    path("data/entry/", pages.DataEntryView.as_view(),   name="data_entry"),    # add / import / edit / delete
    path("activity/",   pages.ActivityLogView.as_view(), name="activity"),      # audit log
    path("profile/",    pages.ProfileView.as_view(),     name="profile"),
    path("live/", views_live.live_page, name="live"),
    path("live/tick", views_live.live_tick, name="live_tick"),
    path("live/retrain", views_live.live_retrain, name="live_retrain"),

    # Streaming pipeline on the separate worker container (web ↔ redis ↔ worker)
    path("pipeline/start",  views_pipeline.pipeline_start,  name="pipeline_start"),
    path("pipeline/stop",   views_pipeline.pipeline_stop,   name="pipeline_stop"),
    path("pipeline/status", views_pipeline.pipeline_status, name="pipeline_status"),

    # Auth
    path("login/",
         auth_views.LoginView.as_view(template_name="anomaly/login.html"),
         name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("register/", av.RegisterView.as_view(), name="register"),
]