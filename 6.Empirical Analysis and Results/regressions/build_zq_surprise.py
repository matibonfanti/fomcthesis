#!/usr/bin/env python3
"""
build_zq_surprise.py
--------------------
Compute a per-meeting ZQ-based monetary-policy surprise.

Inputs (read-only):
  • S3 ticks: s3://fomcthesiss3/market-data/es_ticks/ZQ/date=YYYY-MM-DD/YYYY-MM-DD_ZQ_trades.parquet
  • CSV map : /home/ubuntu/fomc_video_dates.csv   with columns: video_id,meeting_date  (YYYY-MM-DD)

Method:
  • Event time t0 = 14:30 US/Eastern on meeting_date (keeps your current alignment rule).
  • Use ZQ meeting-month contract (e.g., Nov=‘X’), fallback to next month if no trades in window.
  • Δ implied rate (bps) = (100 - P_post) - (100 - P_pre) = P_pre - P_post, over ±10 min window.
  • Scale to target-change bps:   Δ* × (days_in_month / days_remaining_after_announcement)
     where days_remaining_after_announcement = (last day of month - day_of_meeting + 1).
Outputs (local only):
  • /home/ubuntu/reg_outputs/results/zq_surprises.csv
  • /home/ubuntu/reg_outputs/results/zq_surprises_summary.txt
"""

import os, sys, math, json
from pathlib import Path
from datetime import datetime, timedelta, date
from calendar import monthrange

import numpy as np
import pandas as pd
import s3fs
from dateutil import tz

# ---------------------------- config ---------------------------------
BUCKET = "fomcthesiss3"
S3_PATH_TMPL = "market-data/es_ticks/ZQ/date={d}/{d}_ZQ_trades.parquet"

MAP_CSV = "/home/ubuntu/fomc_video_dates.csv"

OUT_DIR = Path("/home/ubuntu/reg_outputs/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "zq_surprises.csv"
OUT_SUM = OUT_DIR / "zq_surprises_summary.txt"

# event window (seconds)
PRE_SEC  = 10 * 60     # 10 minutes before
POST_SEC = 10 * 60     # 10 minutes after

LOCAL_TZ = tz.gettz("US/Eastern")

MONTH_CODE = {1:"F",2:"G",3:"H",4:"J",5:"K",6:"M",7:"N",8:"Q",9:"U",10:"V",11:"X",12:"Z"}

# ---------------------------------------------------------------------

def t0_utc_on_meeting_day(meeting_date_str: str) -> pd.Timestamp:
    """Return event time (14:30 ET) as tz-aware UTC Timestamp."""
    dt_local = datetime.strptime(meeting_date_str, "%Y-%m-%d").replace(hour=14, minute=30, second=0, microsecond=0, tzinfo=LOCAL_TZ)
    return pd.Timestamp(dt_local).tz_convert("UTC")

def zq_symbol_for_meeting(meeting_date: date) -> str:
    """Meeting-month symbol (primary)."""
    m = meeting_date.month
    y = meeting_date.year % 10
    return f"ZQ{MONTH_CODE[m]}{y}"

def zq_next_month_symbol(meeting_date: date) -> str:
    """Next month symbol (fallback)."""
    y = meeting_date.year
    m = meeting_date.month + 1
    if m == 13:
        m, y = 1, y + 1
    return f"ZQ{MONTH_CODE[m]}{y % 10}"

def read_ticks(day_iso: str) -> pd.DataFrame:
    """Read ZQ ticks parquet for a meeting day (handles tz-aware ts_event)."""
    fs = s3fs.S3FileSystem(anon=False)
    s3p = f"s3://{BUCKET}/" + S3_PATH_TMPL.format(d=day_iso)
    df = pd.read_parquet(s3p, filesystem=fs)

    # Robust ts_event handling
    if pd.api.types.is_integer_dtype(df["ts_event"]):
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    elif pd.api.types.is_datetime64_any_dtype(df["ts_event"]) or pd.api.types.is_datetime64tz_dtype(df["ts_event"]):
        df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)
    else:
        raise TypeError(f"Unexpected ts_event dtype: {df['ts_event'].dtype}")

    # Keep only single-leg monthly contracts like ZQX4 (drop spreads like 'ZQ:BF F5-G5-J5')
    df = df[df["symbol"].str.match(r"^ZQ[A-Z]\d$", na=False)].copy()

    # Ensure sorted
    df = df.sort_values("ts_event")
    return df

def last_price_in_window(series: pd.Series, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> float | None:
    """Return the last observed price in [start_ts, end_ts]."""
    s = series.loc[(series.index >= start_ts) & (series.index <= end_ts)]
    if s.empty:
        return None
    return float(s.iloc[-1])

def compute_for_meeting(day_iso: str, df_day: pd.DataFrame) -> dict:
    """Compute surprise record for a meeting day given full-day ticks."""
    t0 = t0_utc_on_meeting_day(day_iso)
    t_pre_start  = t0 - pd.Timedelta(seconds=PRE_SEC)
    t_post_end   = t0 + pd.Timedelta(seconds=POST_SEC)

    meeting_dt = datetime.strptime(day_iso, "%Y-%m-%d").date()
    pri_sym = zq_symbol_for_meeting(meeting_dt)
    alt_sym = zq_next_month_symbol(meeting_dt)

    out = {
        "meeting_id": day_iso,
        "t0_utc": t0.isoformat(),
        "primary_symbol": pri_sym,
        "fallback_symbol": alt_sym,
        "symbol_used": None,
        "price_pre": np.nan,
        "price_post": np.nan,
        "implied_pre": np.nan,
        "implied_post": np.nan,
        "delta_implied_bps": np.nan,
        "days_in_month": monthrange(meeting_dt.year, meeting_dt.month)[1],
        "days_remaining_after_announcement": None,
        "scaling_factor": None,
        "target_surprise_bps": np.nan,
        "notes": "",
    }

    # days remaining in month including meeting day remainder
    D = out["days_in_month"]
    d = meeting_dt.day
    rem = D - d + 1
    out["days_remaining_after_announcement"] = rem
    out["scaling_factor"] = (D / rem) if rem > 0 else np.nan

    # build a per-symbol price series
    rec = None
    for sym_try in (pri_sym, alt_sym):
        sub = df_day[df_day["symbol"] == sym_try].copy()
        if sub.empty:
            continue
        sub = sub.set_index("ts_event")["price"].sort_index()

        p_pre  = last_price_in_window(sub, t_pre_start, t0 - pd.Timedelta(milliseconds=1))
        p_post = last_price_in_window(sub, t0, t_post_end)

        if p_pre is not None and p_post is not None:
            rec = (sym_try, p_pre, p_post)
            break

    if rec is None:
        out["notes"] = "no trades for meeting-month or next-month symbol in event window"
        return out

    sym_used, p_pre, p_post = rec
    out["symbol_used"] = sym_used
    out["price_pre"] = p_pre
    out["price_post"] = p_post

    # implied rate = 100 - price
    imp_pre  = 100.0 - p_pre
    imp_post = 100.0 - p_post
    out["implied_pre"] = imp_pre
    out["implied_post"] = imp_post

    delta_bps = (imp_post - imp_pre) * 100.0      # change in implied monthly avg, in bps
    out["delta_implied_bps"] = delta_bps

    # scale to target change (approx) using remaining-day factor
    sf = out["scaling_factor"]
    if not (sf is None or (isinstance(sf, float) and (math.isnan(sf) or math.isinf(sf)))):
        out["target_surprise_bps"] = delta_bps * sf

    return out

def main():
    # load meeting list
    m = pd.read_csv(MAP_CSV)
    if not {"video_id", "meeting_date"}.issubset(m.columns):
        sys.exit(f"[error] {MAP_CSV} must have columns: video_id,meeting_date")
    meeting_dates = sorted(m["meeting_date"].dropna().unique().tolist())
    print(f"[info] meetings from {MAP_CSV}: {len(meeting_dates)} rows")

    rows = []
    for day in meeting_dates:
        try:
            df_day = read_ticks(day)
        except FileNotFoundError:
            print(f"[warn] ticks parquet not found for {day}, skipping")
            continue
        except Exception as e:
            print(f"[warn] failed reading ticks for {day}: {e}")
            continue

        rec = compute_for_meeting(day, df_day)
        rows.append(rec)
        sym = rec["symbol_used"] or "None"
        print(f"[ok] {day}: sym_used={sym:>6}  Δm_bps={rec['delta_implied_bps']!s:>8}  target_surprise_bps={rec['target_surprise_bps']!s:>8}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)
    print(f"[write] {OUT_CSV}")

    # human summary
    with open(OUT_SUM, "w") as f:
        f.write("ZQ surprises per meeting (bps)\n")
        f.write("meeting_id,symbol_used,Δimplied_month_bps,target_surprise_bps\n")
        for r in rows:
            f.write(f"{r['meeting_id']},{r['symbol_used']},{r['delta_implied_bps']},{r['target_surprise_bps']}\n")
    print(f"[write] {OUT_SUM}\n[done]")

if __name__ == "__main__":
    main()