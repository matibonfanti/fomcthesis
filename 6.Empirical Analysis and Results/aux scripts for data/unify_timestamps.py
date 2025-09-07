#!/usr/bin/env python3
"""
make_segments_parquet.py
Turns every *_inference.json into one Parquet with true UTC epoch seconds.
Assumes each YouTube video starts exactly 14:30 Eastern.
"""

import re, json, s3fs, pyarrow as pa, pyarrow.parquet as pq
from datetime import datetime, time, timedelta
from dateutil import tz

BUCKET      = "fomcthesiss3"
RAW_PREFIX  = "fomc_analysis/08_inference_results"
OUT_PATH    = "s3://fomcthesiss3/analysis/segments.parquet"
LOCAL_TZ    = tz.gettz("US/Eastern")
BATCH_ROWS  = 50_000

FNAME_RE = re.compile(
    r"FOMC_(?P<date>\d{8})_(?P<vid>[A-Za-z0-9_-]{11}).*?_seg(?P<seg>\d+)_inference\.json$"
)

fs = s3fs.S3FileSystem()

def meeting_start_utc(d8):
    """Return datetime UTC for 14:30 Eastern (handles DST) on yyyymmdd."""
    dt_local = datetime.strptime(d8, "%Y%m%d").replace(hour=14, minute=30, tzinfo=LOCAL_TZ)
    return dt_local.astimezone(tz.UTC)

schema = pa.schema({
    "meeting_id": pa.string(),
    "video_id": pa.string(),
    "segment_id": pa.int32(),
    "timestamp_utc": pa.int64(),
    "emotion": pa.string(),
    "is_non_neutral": pa.int8(),
})

writer = pq.ParquetWriter(OUT_PATH, schema, filesystem=fs, compression="zstd")
batch, total = [], 0

for dirpath, _, filenames in fs.walk(f"{BUCKET}/{RAW_PREFIX}"):
    for fn in filenames:
        if not fn.endswith("_inference.json"):
            continue
        m = FNAME_RE.match(fn)
        if not m:
            continue

        start_utc = meeting_start_utc(m["date"])
        with fs.open(f"s3://{dirpath}/{fn}") as f:
            js = json.load(f)

        offset = js["input_segment_info"]["segment_start_s"]
        ts_epoch = int((start_utc + timedelta(seconds=offset)).timestamp())

        row = {
            "meeting_id": f"{m['date'][:4]}-{m['date'][4:6]}-{m['date'][6:]}",
            "video_id": m["vid"],
            "segment_id": int(m["seg"]),
            "timestamp_utc": ts_epoch,
            "emotion": js["inference_details"]["parsed_answer"],
            "is_non_neutral": int(js["inference_details"]["parsed_answer"].lower() != "neutral"),
        }
        batch.append(row)

        if len(batch) >= BATCH_ROWS:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
            total += len(batch); batch.clear()

if batch:
    writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    total += len(batch)

writer.close()
print(f"✓ {total:,} rows written to {OUT_PATH}")
