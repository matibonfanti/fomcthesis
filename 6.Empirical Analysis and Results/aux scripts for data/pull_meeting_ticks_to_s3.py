#!/usr/bin/env python3
"""
Download CME futures ticks for each FOMC meeting day listed in
fomc_video_dates.csv and upload them to S3.
"""

import os, sys, csv, pathlib, boto3, databento as db
import pandas as pd

# ── CONFIG ────────────────────────────────────────────────────────── #
BUCKET   = "fomcthesiss3"
PREFIX   = "market-data/es_ticks"              # no slashes at ends
SYMS     = ["ES.FUT", "ZT.FUT"]                # NEW ➜ parent symbols
DATASET  = "GLBX.MDP3"
SCHEMA   = "trades"
START_HH = "08:30:00"      # UTC
END_HH   = "22:00:00"      # UTC
CSV_MAP  = "fomc_video_dates.csv"
# ──────────────────────────────────────────────────────────────────── #

s3      = boto3.client("s3")
client  = db.Historical(os.environ["DATABENTO_API_KEY"])

def s3_key(sym, day):
    root = sym.split(".")[0]                   # ES or TU
    fname = f"{day}_{root}_{SCHEMA}.parquet"
    return f"{PREFIX}/{root}/date={day}/{fname}"

def already_uploaded(key):
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False

def fetch_and_upload(sym, day):
    key = s3_key(sym, day)
    if already_uploaded(key):
        print(f"↷  Skip {sym} {day} (already in S3)")
        return

    start = f"{day}T{START_HH}Z"
    end   = f"{day}T{END_HH}Z"
    print(f"⇩  Pulling {sym} {day} … ", end="", flush=True)

    resp = client.timeseries.get_range(
        dataset   = DATASET,
        symbols   = SYM,
        stype_in  = "parent",
        schema    = SCHEMA,
        start     = start,
        end       = end,
        format    = "parquet",    # ← works on v0.14
        compression = "zstd",
    )

    df = resp.to_df()
    if df.empty:
        print("no rows — skipped")
        return

    s3_uri = f"s3://{BUCKET}/{key}"
    df.to_parquet(s3_uri, index=False, compression="zstd")
    print(f"{len(df):,} rows → {s3_uri}")

def main():
    dates = [
        row[1] for row in csv.reader(open(CSV_MAP))
        if row and row[0] != "video_id"
    ]
    if not dates:
        sys.exit("No dates found in mapping CSV.")

    for day in sorted(set(dates)):
        for sym in SYMS:
            fetch_and_upload(sym, day)

if __name__ == "__main__":
    main()
