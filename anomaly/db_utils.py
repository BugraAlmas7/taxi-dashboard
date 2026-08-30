from django.db import connection
import pandas as pd


def sql_df(query, params=None):
    """Run a SQL query and return a DataFrame."""
    with connection.cursor() as cur:
        cur.execute(query, params or [])
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)