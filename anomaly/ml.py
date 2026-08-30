import polars as pl, pandas as pd, numpy as np, lightgbm as lgb, holidays
TATIL = holidays.US(years=range(2014,2020))

def gunluk_metrik(parquet_yol, metrik, vendor="hepsi", bas="2015-01-01", son="2016-12-31"):
    lf=(pl.scan_parquet(parquet_yol)
          .with_columns(((pl.col("tpep_dropoff_datetime")-pl.col("tpep_pickup_datetime")).dt.total_seconds()/60).alias("sure_dk"))
          .with_columns(pl.when(pl.col("sure_dk")>0).then(pl.col("trip_distance")/(pl.col("sure_dk")/60)).otherwise(None).alias("hiz_mph"))
          .with_columns(pl.col("tpep_pickup_datetime").dt.date().alias("gun"))
          .filter((pl.col("gun")>=pd.Timestamp(bas).date()) & (pl.col("gun")<=pd.Timestamp(son).date())))
    if vendor!="hepsi":
        lf=lf.filter(pl.col("VendorID")==int(vendor))
    agg=pl.len() if metrik=="sefer" else pl.col(metrik).median()
    g=lf.group_by("gun").agg(agg.alias("d")).sort("gun").collect(engine="streaming")
    s=pd.Series(g["d"].to_list(), index=pd.to_datetime(g["gun"].to_list())).astype(float)
    return s.reindex(pd.date_range(bas, son, freq="D")).interpolate().bfill().ffill()

def _oz(full,tar):
    r=[]
    for t in tar:
        r.append({"l364":full.get(t-pd.Timedelta(days=364),np.nan),"l365":full.get(t-pd.Timedelta(days=365),np.nan),
                  "l371":full.get(t-pd.Timedelta(days=371),np.nan),"ogy":full.loc[t-pd.Timedelta(days=395):t-pd.Timedelta(days=365)].mean(),
                  "hg":t.dayofweek,"ay":t.month,"g":t.day,"hs":int(t.dayofweek>=5),"tatil":int(t.date() in TATIL)})
        return pd.DataFrame(r, index=tar)

def anomali(TEMIZ, HAM17, metrik, vendor="hepsi", alpha=0.5, CAL=120):
    s=gunluk_metrik(TEMIZ, metrik, vendor)
    g17=gunluk_metrik(HAM17, metrik, vendor, "2017-01-01","2017-12-31")
    eg2=s.iloc[:-CAL]
    e2=eg2.index[eg2.index>=eg2.index.min()+pd.Timedelta(days=395)]
    mc=lgb.LGBMRegressor(n_estimators=400, num_leaves=31,learning_rate=0.05,random_state=42,verbose=-1).fit(_oz(eg2,e2),eg2.loc[e2])
    q=np.quantile(np.abs(s.values[-CAL:]-mc.predict(_oz(eg2,s.index[-CAL:]))),1-alpha)

    eg=s.index[s.index>=s.index.min()+pd.Timedelta(days=395)]
    m=lgb.LGBMRegressor(n_estimators=400, num_leaves=31,learning_rate=0.05, random_state=42, verbose=-1).fit(_oz(s,eg), s.loc[eg])
    tah= pd.Series(m.predict(_oz(s,g17.index)),index=g17.index)
    anom=(g17-tah).abs()>q
    return  g17, tah, q, anom

def gun_ici(HAM, bas, son, metrik, vendor="hepsi", coz="1h", k=3.5):
    b = pd.Timestamp(bas).normalize(); s = pd.Timestamp(son).normalize()
    lf = (pl.scan_parquet(HAM)
          .with_columns(((pl.col("tpep_dropoff_datetime")-pl.col("tpep_pickup_datetime")).dt.total_seconds()/60).alias("sure_dk"))
          .with_columns(pl.when(pl.col("sure_dk")>0).then(pl.col("trip_distance")/(pl.col("sure_dk")/60)).otherwise(None).alias("hiz_mph"))
          .with_columns(pl.col("tpep_pickup_datetime").dt.truncate(coz).alias("zaman")))
    if vendor != "hepsi":
        lf = lf.filter(pl.col("VendorID") == int(vendor))
    agg = pl.len() if metrik=="sefer" else pl.col(metrik).mean()
    df = lf.group_by("zaman").agg(agg.alias("d")).sort("zaman").collect(engine="streaming").to_pandas()
    df["zaman"] = pd.to_datetime(df["zaman"])
    df["wd"] = df["zaman"].dt.dayofweek           
    df["tb"] = df["zaman"].dt.time                
    grp = df.groupby(["wd","tb"])["d"]            
    df["bek"] = grp.transform("median")
    df["mad"] = grp.transform(lambda x:(x-x.median()).abs().median()).replace(0, np.nan)
    df["z"]   = 0.6745*(df["d"]-df["bek"])/df["mad"]
    df["anom"] = df["z"].abs() > k
    g = df[(df["zaman"].dt.normalize() >= b) & (df["zaman"].dt.normalize() <= s)].copy()
    return g