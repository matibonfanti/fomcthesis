#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_clip_level_regression.py
============================
Simple clip-level regressions (no valence split; just Non-neutral vs Neutral),
clustered by meeting_id, with an optional winsorization of the dependent var.

Inputs (Parquet; S3 or local):
  - segments: meeting_id, segment_id, timestamp_utc, emotion  (video_id optional)
  - returns : meeting_id, segment_id, timestamp_utc, sym, d_px_60, pre_px_60, (is_non_neutral optional)

Defaults (changeable via CLI flags):
  --segments s3://fomcthesiss3/analysis/segments.parquet
  --returns  s3://fomcthesiss3/analysis/seg_returns.parquet
  --outdir   ~/reg_outputs

Outputs (LOCAL ONLY → <outdir>/results):
  - clip_reg_base.csv    : tidy coefficients (coef, se, pval) for pooled/ES/ZT
  - clip_reg_base.txt    : readable summaries + diagnostics

Run:
  chmod +x run_clip_level_regression.py
  ./run_clip_level_regression.py
  # or with explicit paths:
  ./run_clip_level_regression.py --segments /path/to/segments.parquet --returns /path/to/seg_returns.parquet --outdir /mnt/data/reg_outputs

Notes:
  - The model is:  d_px_60_w ~ is_non_neutral + pre_px_60 + C(sym)   (clustered by meeting_id)
  - ES and ZT units differ; C(sym) absorbs a level shift in the pooled spec.
  - If returns lacks is_non_neutral but has emotion, we derive it: 1{emotion != 'neutral'}.
"""

import argparse, sys, json, textwrap
from pathlib import Path

# ---- dependency checks (we don't auto-install to keep it predictable) ----
missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas")
try:
    import pyarrow  # noqa
except Exception:
    missing.append("pyarrow")
try:
    import statsmodels.formula.api as smf
    from statsmodels.iolib.summary2 import summary_col
except Exception:
    missing.append("statsmodels")
try:
    import s3fs  # noqa  # only needed if you pass s3:// URIs
    _HAS_S3 = True
except Exception:
    _HAS_S3 = False

if missing:
    print("[fatal] missing packages:", ", ".join(missing))
    print("Install with:\n  pip install --user " + " ".join(missing))
    sys.exit(1)

import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.contrast import ContrastResults

# -------------------- helpers --------------------
def is_s3(uri: str) -> bool:
    return isinstance(uri, str) and uri.lower().startswith("s3://")

def read_parquet(uri: str) -> pd.DataFrame:
    if is_s3(uri):
        if not _HAS_S3:
            print("[fatal] s3:// path provided but s3fs not installed. Run: pip install --user s3fs")
            sys.exit(1)
        import s3fs  # type: ignore
        fs = s3fs.S3FileSystem(anon=False)
        return pd.read_parquet(uri, filesystem=fs)
    else:
        return pd.read_parquet(uri)

def winsorize(s: pd.Series, sigma: float) -> pd.Series:
    s = s.copy()
    if sigma and pd.api.types.is_numeric_dtype(s):
        cap = sigma * s.std(ddof=0)
        return s.clip(-cap, cap)
    return s

def tidy_coefs(fit, model_id: str) -> pd.DataFrame:
    out = (
        fit.params.to_frame("coef")
        .join(fit.bse.rename("se"))
        .join(fit.pvalues.rename("pval"))
        .reset_index()
        .rename(columns={"index": "term"})
    )
    out.insert(0, "model", model_id)
    out["nobs"] = int(fit.nobs)
    try:
        out["r2"] = float(fit.rsquared)
    except Exception:
        out["r2"] = pd.NA
    return out

def wald_line(fit, hypothesis: str, label: str) -> str:
    try:
        res: ContrastResults = fit.f_test(hypothesis)
        return f"{label}: F({int(res.df_num)},{int(res.df_denom)})={res.fvalue[0][0]:.3f}, p={res.pvalue:.4f}"
    except Exception as e:
        return f"{label}: (failed: {e})"

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Clip-level regression (local outputs only).")
    ap.add_argument("--segments", default="s3://fomcthesiss3/analysis/segments.parquet",
                    help="Path/URI to segments.parquet")
    ap.add_argument("--returns",  default="s3://fomcthesiss3/analysis/seg_returns.parquet",
                    help="Path/URI to seg_returns.parquet")
    ap.add_argument("--outdir",   default=str(Path.home() / "reg_outputs"),
                    help="Local directory for outputs (default: ~/reg_outputs)")
    ap.add_argument("--winsor-sigma", type=float, default=3.0,
                    help="Winsorize d_px_60 at ±sigma (default 3.0; set 0 to disable)")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] reading segments: {args.segments}")
    segments = read_parquet(args.segments)
    print(f"[info] reading returns : {args.returns}")
    returns = read_parquet(args.returns)

    # Required columns (video_id optional)
    need_seg = {"meeting_id","segment_id","timestamp_utc","emotion"}
    need_ret = {"meeting_id","segment_id","timestamp_utc","sym","d_px_60","pre_px_60"}
    miss_seg = need_seg - set(segments.columns)
    miss_ret = need_ret - set(returns.columns)
    if miss_seg:
        sys.exit(f"[fatal] segments missing columns: {miss_seg}")
    if miss_ret:
        sys.exit(f"[fatal] returns missing columns: {miss_ret}")

    # Merge emotion/is_non_neutral if needed
    segments["emotion"] = segments["emotion"].astype(str).str.lower().str.strip()
    if "is_non_neutral" not in returns.columns:
        if "emotion" not in returns.columns:
            # attach emotion by (meeting_id, segment_id) m:1
            returns = returns.merge(
                segments[["meeting_id","segment_id","emotion"]],
                on=["meeting_id","segment_id"], how="left", validate="m:1"
            )
        returns["is_non_neutral"] = (returns["emotion"].fillna("").str.lower() != "neutral").astype(int)
    else:
        # ensure 0/1 ints
        returns["is_non_neutral"] = returns["is_non_neutral"].astype(int)

    # Build dependent var (winsorized copy, non-destructive)
    y = winsorize(returns["d_px_60"], args.winsor_sigma)
    returns = returns.assign(d_px_60_w=y)

    # Drop rows with missing y or controls
    df = returns.dropna(subset=["d_px_60_w","pre_px_60","is_non_neutral","meeting_id","sym"]).copy()

    # meeting_id as category (but clustering by original labels is fine)
    df["meeting_id"] = df["meeting_id"].astype(str)

    # Three models: pooled (ES+ZT), ES only, ZT only
    outputs = []
    summaries = []
    models = [
        ("pooled", df),
        ("ES_only", df[df["sym"]=="ES"]),
        ("ZT_only", df[df["sym"]=="ZT"]),
    ]

    for model_id, d in models:
        if d.empty or d["meeting_id"].nunique() < 2:
            summaries.append(f"[warn] {model_id}: insufficient data (rows={len(d)}, unique meetings={d['meeting_id'].nunique()}) — skipped.")
            continue

        # Formula: pooled has C(sym); single-asset drops it
        formula = "d_px_60_w ~ is_non_neutral + pre_px_60" + ("" if model_id!="pooled" else " + C(sym)")

        fit = smf.ols(formula, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["meeting_id"]}
        )

        outputs.append(tidy_coefs(fit, model_id))

        # text summary block
        head = f"\n=== {model_id} ===\nrows={len(d):,}, meetings={d['meeting_id'].nunique()}, winsor_sigma={args.winsor_sigma}"
        base = fit.summary().as_text()
        wald = wald_line(fit, "is_non_neutral = 0", "Wald test (β_nonneutral = 0)")
        summaries.append("\n".join([head, base, wald]))

    if not outputs:
        sys.exit("[fatal] no estimable models; check inputs.")

    coef_all = pd.concat(outputs, ignore_index=True)
    coef_path = results_dir / "clip_reg_base.csv"
    coef_all.to_csv(coef_path, index=False)
    print(f"[write] {coef_path}")

    # write human-readable summaries
    txt = [
        "Clip-level regression (Non-neutral vs Neutral)",
        f"segments = {args.segments}",
        f"returns  = {args.returns}",
        f"winsor_sigma = ±{args.winsor_sigma}",
        "",
        *summaries
    ]
    txt_path = results_dir / "clip_reg_base.txt"
    txt_path.write_text("\n".join(txt), encoding="utf-8")
    print(f"[write] {txt_path}")

    # small diagnostics JSON
    diag = {
        "n_rows_input": int(len(returns)),
        "n_rows_used": int(len(df)),
        "unique_meetings": int(df["meeting_id"].nunique()),
        "sym_counts": df["sym"].value_counts(dropna=False).to_dict(),
    }
    (results_dir / "diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"[write] {results_dir / 'diagnostics.json'}")

    print("\n[done] Regressions finished. See:", results_dir)

if __name__ == "__main__":
    main()