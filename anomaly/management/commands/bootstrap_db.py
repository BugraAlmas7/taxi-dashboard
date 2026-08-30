"""
bootstrap_db — create every anomaly-app table on a FRESH database.

Why this exists: the trip tables (`sefer_egitim`, `sefer_2017`) and the
profile/cache tables were always created OUTSIDE Django (by the Colab
duckdb/upload scripts). Their models were generated as `managed = False`, so
`migrate` never creates them — and the later migrations try to ALTER
`sefer_egitim`, which crashes on a fresh DB where the table doesn't exist.

This command sidesteps that: it fake-applies the anomaly migrations (so no
broken ALTER runs), migrates the contrib apps normally, then creates any
missing anomaly table directly with the schema editor. Idempotent — safe to
run more than once and safe against an existing database (existing tables are
left untouched).

Usage:
    docker compose exec web python manage.py bootstrap_db
"""
from django.apps import apps
from django.core.management import call_command
from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = "Create all anomaly-app tables on a fresh database (safe to re-run)."

    def _exists(self, table):
        with connection.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", [table])
            return cur.fetchone()[0] is not None

    def handle(self, *args, **opts):
        # 1) Bring the contrib apps up first (auth is needed for the FKs below).
        self.stdout.write("Migrating contenttypes / auth ...")
        call_command("migrate", "contenttypes", verbosity=0)
        call_command("migrate", "auth", verbosity=0)

        # 2) Fake-apply the anomaly migrations so their broken ALTERs never run.
        self.stdout.write("Fake-applying anomaly migrations ...")
        call_command("migrate", "anomaly", fake=True, verbosity=0)

        # 3) Migrate everything else (admin, sessions, ...).
        self.stdout.write("Migrating remaining apps ...")
        call_command("migrate", verbosity=0)

        # 4) Create any missing anomaly table directly.
        self.stdout.write("Creating anomaly tables ...")
        created = 0
        with connection.schema_editor() as se:
            for model in apps.get_app_config("anomaly").get_models():
                table = model._meta.db_table
                if self._exists(table):
                    self.stdout.write(f"  ok  {table} (exists)")
                    continue
                se.create_model(model)
                created += 1
                self.stdout.write(self.style.SUCCESS(f"  +   {table} created"))

        # 5) Sync columns on EXISTING tables: if a model gained a field after its
        #    table was first created (e.g. manual_entry.trip_duration_minutes),
        #    add the missing column via Django's schema editor — pure ORM, no
        #    hand-written SQL. Existing columns are never touched or dropped.
        self.stdout.write("Syncing missing columns ...")
        added = 0
        for model in apps.get_app_config("anomaly").get_models():
            table = model._meta.db_table
            if not self._exists(table):
                continue  # just created above with the full schema
            with connection.cursor() as cur:
                existing = {
                    col.name
                    for col in connection.introspection.get_table_description(cur, table)
                }
            for field in model._meta.local_fields:
                if field.column in existing:
                    continue
                if not field.null and not field.has_default():
                    # can't safely add a NOT NULL column without a default to a
                    # table that may already hold rows — needs a real migration.
                    self.stdout.write(self.style.WARNING(
                        f"  !!  {table}.{field.column} is missing but NOT NULL "
                        f"with no default — add it with a migration"))
                    continue
                with connection.schema_editor() as se:
                    se.add_field(model, field)
                added += 1
                self.stdout.write(self.style.SUCCESS(f"  +   {table}.{field.column} added"))
        if not added:
            self.stdout.write("  ok  all columns present")

        self.stdout.write(self.style.SUCCESS(
            f"\nDone. {created} table(s) created, {added} column(s) added. "
            f"You can now load data with setup_data.py."))
