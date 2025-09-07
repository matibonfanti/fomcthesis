#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_per_emotion_regressions.py
==============================
Disaggregated per-emotion clip-level regressions at 60s horizon, ES and ZT
separately, with cluster-robust SEs by meeting, estimated:
  (a) without meeting FE
  (b) with meeting FE: C(meeting_id)

Outcome: d_px_60 (winsorized copy ±sigma)
Regressors: emo_happy, emo_surprise, emo_anxious (neutral omitted), pre_px_60
Errors: clustered by meeting_id

Inputs (Parquet; S3 or local):
  --segments: meeting_id, segment_id, timestamp_utc, emotion  (video_id optional)
  --returns : meeting_id, segment_id, timestamp_utc, sym, d_px_60, pre_px_60
              (emotion optional; if absent we merge it in)

Outputs (LOCAL ONLY → <outdir>/results):
  per_emotion_60s_tidy.csv        # tidy coefficients for ES/ZT × {noFE, FE}
  per_emotion_60s_counts.csv      # counts by emotion used in each model
  per_emotion_60s_summary.txt     # human-readable summaries & joint tests

Run:
  chmod +x run_per_emotion_regressions.py
  ./run_per_emotion_regressions.py
  # or with explicit paths:
  ./run_per_emotion_regressions.py \
      --segments s3://fomcthesiss3/analysis/segments.parquet \
      --returns  s3://fomcthesiss3/analysis/seg_returns.parquet \
      --outdir   /mnt/data/reg_outputs
"""

import argparse, sys, json, textwrap
from pathlib import Path

# ---- dependency checks ----
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
    import s3fs  # noqa  # only if you pass s3:// URIs
    _HAS_S3 = True
except Exception:
    _HAS_S3 = False

if missing:
    print("[fatal] missing packages:", ", ".join(missing))
    print("Install with:\n  pip install --user " + " ".join(missing))
    sys.exit(1)

import pandas as pd
import statsmodels.formula.api as smf

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
    if not sigma: return s
    cap = sigma * s.std(ddof=0)
    return s.clip(-cap, cap)

def tidy_coefs(fit, model_id: str, asset: str, spec: str, meetings_used: int) -> pd.DataFrame:
    out = (
        fit.params.to_frame("coef")
        .join(fit.bse.rename("se"))
        .join(fit.pvalues.rename("pval"))
        .reset_index()
        .rename(columns={"index": "term"})
    )
    out.insert(0, "asset", asset)
    out.insert(1, "spec", spec)  # "noFE" or "FE"
    out.insert(2, "model", model_id)
    out["nobs"] = int(fit.nobs)
    out["r2"] = float(getattr(fit, "rsquared", float("nan")))
    out["meetings_used"] = int(meetings_used)
    return out

def ftest_str(fit, hyp: str, label: str) -> str:
    try:
        res = fit.f_test(hyp)
        # robust extraction across statsmodels versions
        fval = float(getattr(res, "fvalue", getattr(res, "statistic", float("nan"))))
        pval = float(getattr(res, "pvalue", getattr(res, "pval", float("nan"))))
        df_num = int(getattr(res, "df_num", 0))
        df_den = int(getattr(res, "df_denom", 0))
        return f"{label}: F({df_num},{df_den})={fval:.3f}, p={pval:.4f}"
    except Exception as e:
        return f"{label}: (failed: {e})"

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="Per-emotion clip-level regressions at 60s (local outputs only).")
    ap.add_argument("--segments", default="s3://fomcthesiss3/analysis/segments.parquet",
                    help="Path/URI to segments.parquet")
    ap.add_argument("--returns",  default="s3://fomcthesiss3/analysis/seg_returns.parquet",
                    help="Path/URI to seg_returns.parquet")
    ap.add_argument("--outdir",   default=str(Path.home() / "reg_outputs"),
                    help="Local directory for outputs (default: ~/reg_outputs)")
    ap.add_argument("--winsor-sigma", type=float, default=3.0,
                    help="Winsorize d_px_60 at ±sigma (default 3.0; set 0 to disable)")
    ap.add_argument("--allowed-labels", nargs="*", default=["neutral","happy","surprise","anxious"],
                    help="Which labels to keep (others dropped)")
    args = ap.parse_args()

    outdir = Path(args.outdir).expanduser().resolve()
    results_dir = outdir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] reading segments: {args.segments}")
    segments = read_parquet(args.segments)
    print(f"[info] reading returns : {args.returns}")
    returns = read_parquet(args.returns)

    # Basic requirements (video_id optional)
    need_seg = {"meeting_id","segment_id","timestamp_utc","emotion"}
    need_ret = {"meeting_id","segment_id","timestamp_utc","sym","d_px_60","pre_px_60"}
    miss_seg = need_seg - set(segments.columns)
    miss_ret = need_ret - set(returns.columns)
    if miss_seg:
        sys.exit(f"[fatal] segments missing columns: {miss_seg}")
    if miss_ret:
        sys.exit(f"[fatal] returns missing columns: {miss_ret}")

    # Clean labels and attach to returns if needed
    segments["emotion"] = segments["emotion"].astype(str).str.lower().str.strip()
    if "emotion" not in returns.columns:
        returns = returns.merge(
            segments[["meeting_id","segment_id","emotion"]],
            on=["meeting_id","segment_id"], how="left", validate="m:1"
        )
    returns["emotion"] = returns["emotion"].astype(str).str.lower().str.strip()

    # Keep only allowed labels
    returns = returns[returns["emotion"].isin(args.allowed-labels if hasattr(args,'allowed-labels') else args.allowed_labels)]  # safety
    # The above line is a safety fallback; proper access:
    returns = returns[returns["emotion"].isin(args.allowed_labels)].copy()

    # Build dummies: neutral is the omitted reference
    for emo in ["happy","surprise","anxious"]:
        col = f"emo_{emo}"
        returns[col] = (returns["emotion"] == emo).astype(int)

    # Winsorized dependent variable
    returns["d_px_60_w"] = winsorize(returns["d_px_60"], args.winsor_sigma)

    # Prepare per-asset data
    returns["meeting_id"] = returns["meeting_id"].astype(str)
    assets = []
    for sym in ("ES","ZT"):
        d = returns[returns["sym"] == sym].dropna(subset=["d_px_60_w","pre_px_60"]).copy()
        d["meetings_used"] = d["meeting_id"].nunique()
        if d.empty or d["meetings_used"].iloc[0] < 2:
            print(f"[warn] {sym}: insufficient data — skipping.")
            continue
        assets.append((sym, d))

    if not assets:
        sys.exit("[fatal] no asset panels to estimate.")

    tidy_rows = []
    lines = []

    # For each asset, estimate:
    # (1) no FE: y ~ happy + surprise + anxious + pre_px_60
    # (2) FE:    y ~ happy + surprise + anxious + pre_px_60 + C(meeting_id)
    for sym, d in assets:
        # counts by emotion
        counts = d["emotion"].value_counts().reindex(["neutral","happy","surprise","anxious"], fill_value=0)
        # small counts table output
        # assemble summary header
        head = [
            f"\n=== {sym} (60s) ===",
            f"rows={len(d):,}, meetings={d['meetings_used'].iloc[0]}, winsor_sigma={args.winsor_sigma}",
            f"counts: neutral={int(counts.get('neutral',0))}, happy={int(counts.get('happy',0))}, "
            f"surprise={int(counts.get('surprise',0))}, anxious={int(counts.get('anxious',0))}",
        ]
        lines.extend(head)

        # ---- (1) no FE ----
        formula_noFE = "d_px_60_w ~ emo_happy + emo_surprise + emo_anxious + pre_px_60"
        fit_noFE = smf.ols(formula_noFE, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["meeting_id"]}
        )
        tidy_rows.append(tidy_coefs(fit_noFE, "per_emotion_60s", sym, "noFE", d["meetings_used"].iloc[0]))

        lines.append("\n[no FE] Model:")
        lines.append(fit_noFE.summary().as_text())
        lines.append(ftest_str(fit_noFE, "emo_happy = emo_surprise = emo_anxious = 0", "Joint test (all emotion β = 0)"))
        lines.append(ftest_str(fit_noFE, "emo_happy = emo_anxious", "Happy = Anxious"))
        lines.append(ftest_str(fit_noFE, "emo_surprise = emo_anxious", "Surprise = Anxious"))
        lines.append(ftest_str(fit_noFE, "emo_happy = emo_surprise", "Happy = Surprise"))

        # ---- (2) meeting FE ----
        formula_FE = "d_px_60_w ~ emo_happy + emo_surprise + emo_anxious + pre_px_60 + C(meeting_id)"
        fit_FE = smf.ols(formula_FE, data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["meeting_id"]}
        )
        tidy_rows.append(tidy_coefs(fit_FE, "per_emotion_60s", sym, "FE", d["meetings_used"].iloc[0]))

        lines.append("\n[with meeting FE] Model:")
        lines.append(fit_FE.summary().as_text())
        lines.append(ftest_str(fit_FE, "emo_happy = emo_surprise = emo_anxious = 0", "Joint test (all emotion β = 0)"))
        lines.append(ftest_str(fit_FE, "emo_happy = emo_anxious", "Happy = Anxious"))
        lines.append(ftest_str(fit_FE, "emo_surprise = emo_anxious", "Surprise = Anxious"))
        lines.append(ftest_str(fit_FE, "emo_happy = emo_surprise", "Happy = Surprise"))

    # Write tidy coefficients
    tidy_df = pd.concat(tidy_rows, ignore_index=True)
    tidy_path = results_dir / "per_emotion_60s_tidy.csv"
    tidy_df.to_csv(tidy_path, index=False)
    print(f"[write] {tidy_path}")

    # Write counts used
    cnt_rows = []
    for sym, d in assets:
        counts = d["emotion"].value_counts().reindex(["neutral","happy","surprise","anxious"], fill_value=0)
        cnt_rows.append({
            "asset": sym,
            "rows": int(len(d)),
            "meetings": int(d["meeting_id"].nunique()),
            "neutral": int(counts.get("neutral",0)),
            "happy": int(counts.get("happy",0)),
            "surprise": int(counts.get("surprise",0)),
            "anxious": int(counts.get("anxious",0)),
        })
    counts_df = pd.DataFrame(cnt_rows)
    counts_path = results_dir / "per_emotion_60s_counts.csv"
    counts_df.to_csv(counts_path, index=False)
    print(f"[write] {counts_path}")

    # Write human-readable summary
    header = [
        "Disaggregated per-emotion regressions at 60s (ES & ZT; neutral baseline)",
        f"segments = {args.segments}",
        f"returns  = {args.returns}",
        f"winsor_sigma = ±{args.winsor_sigma}",
        "",
    ]
    txt = "\n".join(header + lines)
    txt_path = results_dir / "per_emotion_60s_summary.txt"
    txt_path.write_text(txt, encoding="utf-8")
    print(f"[write] {txt_path}")

    # Diagnostics JSON
    diag = {
        "inputs": {"segments": args.segments, "returns": args.returns},
        "winsor_sigma": args.winsor_sigma,
        "assets_estimated": [sym for sym, _ in assets],
        "n_rows_total": int(len(returns)),
    }
    (results_dir / "per_emotion_60s_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print(f"[write] {results_dir / 'per_emotion_60s_diagnostics.json'}")

    print("\n[done] Per-emotion regressions complete. See:", results_dir)

if __name__ == "__main__":
    main()