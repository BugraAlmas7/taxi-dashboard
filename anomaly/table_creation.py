import polars as pl
from sqlalchemy import create_engine

DSN = "postgresql+psycopg2://taksi_user:183462795@localhost:5432/taksi_db"
engine = create_engine(DSN)

# Take the schema from the CSV's first row
import pandas as pd
header = pd.read_csv(r"D:\veri\ham2017-002.csv", nrows=0)
header.iloc[:0].to_sql("sefer_2017", engine, if_exists="replace", index=False)
print("Kolonlar:", list(header.columns))