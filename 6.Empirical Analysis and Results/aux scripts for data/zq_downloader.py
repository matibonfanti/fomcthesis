#!/usr/bin/env python3
"""
zq_downloader.py  –  download ZQ.FUT trades for each FOMC date
and save Parquet to S3, matching the existing ES/ZT directory style.
"""

import os, boto3, databento as db, pandas as pd

BUCKET  = "fomcthesiss3"
PREFIX  = "market-data/es_ticks"
SYM     = "ZQ.FUT"
DATASET = "GLBX.MDP3"
SCHEMA  = "trades"
START_HH = "08:30:00"         # UTC (08:30 ET)
END_HH   = "22:00:00"         # UTC (16:00 ET)

MEETING_DATES = [
    "2019-05-01","2021-11-03","2022-07-27","2022-11-02",
    "2023-03-22","2023-06-14","2023-11-01","2023-12-13",
    "2024-03-20","2024-06-12","2024-09-18","2024-11-07",
    "2024-12-18","2025-01-29","2025-03-19","2025-05-07",
]

s3      = boto3.client("s3")
client  = db.Historical(os.environ["DATABENTO_API_KEY"])

def s3_key(day: str) -> str:
    root  = "ZQ"
    fname = f"{day}_{root}_{SCHEMA}.parquet"
    return f"{PREFIX}/{root}/date={day}/{fname}"

def exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

for day in MEETING_DATES:
    key = s3_key(day)
    if exists(key):
        print(f"↷  Skip {day} (already uploaded)")
        continue

    start = f"{day}T{START_HH}Z"
    end   = f"{day}T{END_HH}Z"
    print(f"⇩  Pulling ZQ {day} …", end="", flush=True)

    # Identical signature to your ES/ZT call—no format/encoding args.
    resp = client.timeseries.get_range(
        dataset   = DATASET,
        symbols   = SYM,
        stype_in  = "parent",
        schema    = SCHEMA,
        start     = start,
        end       = end,
    )

    df = resp.to_df()
    if df.empty:
        print(" no rows → skipped")
        continue

    s3_uri = f"s3://{BUCKET}/{key}"
    df.to_parquet(s3_uri, index=False, compression="zstd")
    print(f" {len(df):,} rows → {s3_uri}")
