#!/usr/bin/env python3
"""
run_reg_with_zq.py
Clip-level per-emotion regressions including ZQ (Fed fund futures) surprise.

Two specs per asset (ES, ZT):
  (A) No meeting FE:  d_px_60_w ~ emotions + pre_px_60 + ZQ_surprise
  (B) With meeting FE: d_px_60_w ~ emotions + pre_px_60 + C(meeting_id)
      (Note: ZQ_surprise drops here by construction, it's meeting-constant.)

Inputs (read-only):
  - s3://fomcthesiss3/analysis/segments.parquet
  - s3://fomcthesiss3/analysis/seg_returns.parquet
  - /home/ubuntu/reg_outputs/results/zq_surprises.csv  (from build_zq_surprise.py)

Outputs (local):
  - /home/ubuntu/reg_outputs/results/per_emotion_60s_with_zq.csv
  - /home/ubuntu/reg_outputs/results/per_emotion_60s_with_zq.txt
"""

import os, sys, json
from pathlib import Path
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

try:
    import s3fs
    FS = s3fs.S3FileSystem(anon=False)
except Exception:
    FS = None

# ---------------- config ----------------
BUCKET      = "fomcthesiss3"
SEG_KEY     = "analysis/segments.parquet"
RET_KEY     = "analysis/seg_returns.parquet"
ZQ_CSV      = "/home/ubuntu/reg_outputs/results/zq_surprises.csv"

OUT_DIR     = Path("/home/ubuntu/reg_outputs/results")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV     = OUT_DIR / "per_emotion_60s_with_zq.csv"
OUT_TXT     = OUT_DIR / "per_emotion_60s_with_zq.txt"

W_SIGMA     = 3.0     # winsor cap (sigma)
ASSETS      = ["ES", "ZT"]
# ----------------------------------------


def _read_pq(s3_key: str) -> pd.DataFrame:
    uri = f"s3://{BUCKET}/{s3_key}"
    print(f"[info] reading {s3_key.split('/')[-1]}: {uri}")
    if FS is None:
        return pd.read_parquet(uri)  # assumes mounted
    return pd.read_parquet(uri, filesystem=FS)

def winsor(s: pd.Series, sigma=W_SIGMA) -> pd.Series:
    cap = sigma * s.std(skipna=True)
    return s.clip(-cap, cap)

def tidy_from_fit(fit, model_name: str, asset: str) -> pd.DataFrame:
    """Return tidy coef table with meta columns."""
    df = (
        fit.params.to_frame("coef")
          .join(fit.bse.rename("se"))
          .join(fit.pvalues.rename("pval"))
          .reset_index()
          .rename(columns={"index":"term"})
    )
    df["model"]      = model_name
    df["asset"]      = asset
    df["nobs"]       = int(fit.nobs)
    df["r2"]         = float(getattr(fit, "rsquared", np.nan))
    return df[["asset","model","term","coef","se","pval","nobs","r2"]]

def main():
    # ---------- load core data ----------
    segments = _read_pq(SEG_KEY)[["meeting_id","video_id","segment_id","emotion"]]
    segments["emotion"] = segments["emotion"].str.lower()

    returns = _read_pq(RET_KEY)
    # merge emotion if missing
    if "emotion" not in returns.columns:
        on_cols = [c for c in ["meeting_id","video_id","segment_id"] if c in returns.columns]
        if "video_id" not in on_cols and "segment_id" in on_cols:
            # common fallback when video_id not present
            on_cols = ["meeting_id","segment_id"]
        returns = returns.merge(
            segments, on=on_cols, how="left", validate="one_to_one"
        )

    # sanity
    needed = {"meeting_id","sym","d_px_60","pre_px_60","emotion"}
    missing = needed - set(returns.columns)
    if missing:
        sys.exit(f"[error] returns missing columns: {missing}")

    # ---------- winsorize y ----------
    returns["d_px_60_w"] = winsor(returns["d_px_60"])

    # ---------- emotion dummies ----------
    emo = returns["emotion"].astype(str).str.lower()
    returns["emo_happy"]    = (emo == "happy").astype(int)
    returns["emo_surprise"] = (emo == "surprise").astype(int)
    returns["emo_anxious"]  = (emo == "anxious").astype(int)
    # neutral is omitted baseline

    # ---------- ZQ surprises ----------
    if not Path(ZQ_CSV).is_file():
        sys.exit(f"[error] ZQ surprises not found: {ZQ_CSV}\nRun build_zq_surprise.py first.")
    zq = pd.read_csv(ZQ_CSV)
    # Expect columns: meeting_id, target_surprise_bps (and others)
    if "meeting_id" not in zq.columns:
        sys.exit("[error] zq_surprises.csv missing 'meeting_id'")
    if "target_surprise_bps" not in zq.columns:
        sys.exit("[error] zq_surprises.csv missing 'target_surprise_bps'")

    zq = zq[["meeting_id","target_surprise_bps"]].copy()

    df = returns.merge(zq, on="meeting_id", how="left")
    # -------------------------------

    all_tidy = []
    with open(OUT_TXT, "w") as fh:
        fh.write("Per-emotion regressions @60s with ZQ surprise control\n")
        fh.write(f"winsor_sigma = ±{W_SIGMA}\n")
        fh.write(f"rows total (pre-dropna) = {len(df):,}\n\n")

        for asset in ASSETS:
            sub = df.loc[df["sym"] == asset].copy()

            # Spec (A): no FE, include ZQ
            cols_a = [
                "d_px_60_w","pre_px_60","meeting_id",
                "emo_happy","emo_surprise","emo_anxious",
                "target_surprise_bps",
            ]
            A = sub[cols_a].dropna().copy()

            # Spec (B): meeting FE, exclude ZQ (collinear with FE)
            cols_b = [
                "d_px_60_w","pre_px_60","meeting_id",
                "emo_happy","emo_surprise","emo_anxious",
            ]
            B = sub[cols_b].dropna().copy()

            # Quick diagnostics
            fh.write(f"=== {asset} ===\n")
            fh.write(f"[A noFE] used rows = {len(A):,} (dropped {len(sub)-len(A):,})\n")
            fh.write(f"  clusters (meetings) = {A['meeting_id'].nunique()}\n")
            fh.write(f"[B   FE] used rows = {len(B):,} (dropped {len(sub)-len(B):,})\n")
            fh.write(f"  clusters (meetings) = {B['meeting_id'].nunique()}\n")

            # (A) No FE + ZQ
            if len(A) == 0 or A["meeting_id"].nunique() < 2:
                fh.write("[warn] insufficient rows or clusters for spec A\n")
            else:
                model_a = smf.ols(
                    "d_px_60_w ~ emo_happy + emo_surprise + emo_anxious + pre_px_60 + target_surprise_bps",
                    data=A
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": A["meeting_id"]}
                )
                fh.write("\n[A] No FE + ZQ surprise\n")
                fh.write(str(model_a.summary()))
                fh.write("\n")
                tidy_a = tidy_from_fit(model_a, "noFE_plus_ZQ", asset)
                all_tidy.append(tidy_a)

            # (B) Meeting FE (ZQ excluded)
            if len(B) == 0 or B["meeting_id"].nunique() < 2:
                fh.write("[warn] insufficient rows or clusters for spec B\n\n")
            else:
                model_b = smf.ols(
                    "d_px_60_w ~ emo_happy + emo_surprise + emo_anxious + pre_px_60 + C(meeting_id)",
                    data=B
                ).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": B["meeting_id"]}
                )
                fh.write("\n[B] With meeting FE (ZQ excluded)\n")
                fh.write(str(model_b.summary()))
                fh.write("\n")
                tidy_b = tidy_from_fit(model_b, "FE_no_ZQ", asset)
                all_tidy.append(tidy_b)

            fh.write("\n")

    if all_tidy:
        out = pd.concat(all_tidy, ignore_index=True)
        out.to_csv(OUT_CSV, index=False)
        print(f"[write] {OUT_CSV}")
    print(f"[write] {OUT_TXT}")
    print("\n[done] Regressions with ZQ surprise completed.")

if __name__ == "__main__":
    main()