#!/usr/bin/env python3
"""
Map each YouTube-ID-named folder under 08_inference_results/ to its video date
using yt-dlp *with browser cookies* – no YouTube API key required.
"""

import os, csv, sys, pathlib, boto3, re
from datetime import datetime
from dateutil import tz
from tabulate import tabulate
from yt_dlp import YoutubeDL

# ───────────────────────────── CONFIG ───────────────────────────── #
MODE        = "s3"      # "local"  -> LOCAL_DIR is scanned
                        # "s3"     -> S3_BUCKET + S3_PREFIX are scanned
LOCAL_DIR   = "/mnt/fomc_analysis/08_inference_results/"
S3_BUCKET   = "fomcthesiss3"
S3_PREFIX   = "fomc_analysis/08_inference_results/"
COOKIE_FILE = "/home/ubuntu/youtube.txt"   # <── your uploaded file
CSV_OUT     = "fomc_video_dates.csv"
# ─────────────────────────────────────────────────────────────────── #

def list_ids_local(base):
    return sorted([p.name for p in pathlib.Path(base).iterdir() if p.is_dir()])

def list_ids_s3(bucket, prefix):
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")
    ids = []
    for page in pages:
        ids += [cp["Prefix"].split("/")[-2]          # last non-empty token
                for cp in page.get("CommonPrefixes", [])]
    return sorted(ids)

def fetch_date(video_id, ydl):
    """Return ISO date string (YYYY-MM-DD) or 'unknown'."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        info = ydl.extract_info(url, download=False)
    except Exception:
        return "unknown"

    # 1️⃣ ordinary uploads
    ud = info.get("upload_date")
    if ud and re.fullmatch(r"\d{8}", ud):
        return datetime.strptime(ud, "%Y%m%d").date().isoformat()

    # 2️⃣ live / premiere
    for field in ("release_timestamp", "timestamp"):
        ts = info.get(field)
        if ts:
            return datetime.utcfromtimestamp(ts).date().isoformat()

    return "unknown"

def main():
    # verify cookiefile exists
    if not pathlib.Path(COOKIE_FILE).is_file():
        sys.exit(f"Cookie file not found: {COOKIE_FILE}")

    if MODE == "local":
        video_ids = list_ids_local(LOCAL_DIR)
    elif MODE == "s3":
        video_ids = list_ids_s3(S3_BUCKET, S3_PREFIX)
    else:
        sys.exit("MODE must be 'local' or 's3'")

    if not video_ids:
        sys.exit("No video-ID folders found for the given prefix.")

    ydl = YoutubeDL({
        "cookiefile": COOKIE_FILE,
        "skip_download": True,
        "quiet": True,
    })

    rows = [(vid, fetch_date(vid, ydl)) for vid in video_ids]

    # write CSV
    with open(CSV_OUT, "w", newline="") as f:
        csv.writer(f).writerows([("video_id", "meeting_date"), *rows])

    # pretty print
    print(tabulate(rows, headers=["YouTube ID", "Date"], tablefmt="github"))
    print(f"\nSaved → {CSV_OUT}  ({len(rows)} rows)")

if __name__ == "__main__":
    main()
