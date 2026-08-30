# ── For the Live page: add to taxi_web/urls.py ──────────────────────────────
# Next to the imports at the top:
from anomaly import views_live          # noqa

# Add to the urlpatterns list (among the other paths):
#     path("live/",     views_live.live_page, name="live"),
#     path("live/tick", views_live.live_tick, name="live_tick"),
#
# NOTE: the fetch "tick" (relative) inside live.html resolves to /live/tick
# because the page lives under /live/. The route "live/tick" matches it exactly.
#
# To add the "Live" link to the nav, put this line in each page's topbar <nav>:
#     <a href="{% url 'live' %}">Live</a>
# (panel.html, raw_data, data_entry and profile each have their own nav — there
#  is no shared base template, so you must add the link to each one individually.)
