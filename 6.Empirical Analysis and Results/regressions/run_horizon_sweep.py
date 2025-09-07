#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_horizon_sweep_v2.py
Dense horizon sweep (5s steps up to 300s) for per-emotion clip-level regressions.
- Reads:   s3://fomcthesiss3/analysis/segments.parquet
           s3://fomcthesiss3/analysis/seg_returns.parquet
           s3://fomcthesiss3/market-data/es_ticks/<SYM>/date=YYYY-MM-DD/*.parquet
- Writes:  local only → /home/ubuntu/reg_outputs/horizon/{results,figures}

Spec: Δp_h ~ emo_happy + emo_surprise + emo_anxious + pre_px_60
Errors clustered by meeting_id. Neutral is omitted category.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import s3fs
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ── CONFIG ──────────────────────────────────────────────────────────────────
BUCKET        = "fomcthesiss3"
SEGMENTS_URI  = f"s3://{BUCKET}/analysis/segments.parquet"
RETURNS_URI   = f"s3://{BUCKET}/analysis/seg_returns.parquet"
TICK_ROOT     = f"s3://{BUCKET}/market-data/es_ticks"   # ES/ZT/date=YYYY-MM-DD/*.parquet

ASSETS        = ["ES", "ZT"]
HORIZONS      = list(range(0, 301, 25))                  # every 5s up to 300s
WINSOR_SIGMA  = 3.0

OUT_ROOT      = "/home/ubuntu/reg_outputs/horizon"
OUT_RESULTS   = os.path.join(OUT_ROOT, "results")
OUT_FIGS      = os.path.join(OUT_ROOT, "figures")
os.makedirs(OUT_RESULTS, exist_ok=True)
os.makedirs(OUT_FIGS, exist_ok=True)

pd.options.display.width = 160
warnings.filterwarnings("ignore", category=FutureWarning)

# ── IO helpers ──────────────────────────────────────────────────────────────
fs = s3fs.S3FileSystem(anon=False)

def read_parquet(uri: str) -> pd.DataFrame:
    return pd.read_parquet(uri, filesystem=fs)

def glob_first(path_glob: str) -> str | None:
    matches = fs.glob(path_glob)
    return matches[0] if matches else None

# ── Price cache builder (per meeting × asset) ───────────────────────────────
def build_price_cache(meeting_ids):
    """dict[(meeting_id, sym)] -> 1s last-trade pd.Series indexed by epoch seconds."""
    cache = {}
    for sym in ASSETS:
        for mtg in sorted(set(meeting_ids)):
            folder = f"{TICK_ROOT}/{sym}/date={mtg}/*.parquet"
            key = glob_first(folder)
            if key is None:
                print(f"[warn] no ticks for {sym} {mtg} under {folder}")
                continue
            ticks = pd.read_parquet(f"s3://{key}", filesystem=fs)
            if "ts_event" not in ticks.columns or "price" not in ticks.columns:
                print(f"[warn] missing ts_event/price in {key}; skipping")
                continue
            sec = ticks["ts_event"].astype("int64") // 1_000_000_000
            price = ticks["price"].astype("float64").copy()
            if sym == "ES":  # keep consistent with earlier scaling
                price = price / 100.0
            series = (pd.DataFrame({"sec": sec, "price": price})
                        .drop_duplicates("sec").sort_values("sec")
                        .set_index("sec")["price"])
            cache[(mtg, sym)] = series
    return cache

# ── Forward returns ─────────────────────────────────────────────────────────
def sec_of(x) -> int:
    if isinstance(x, (int, np.integer)):
        return int(x)
    return int(pd.Timestamp(x).value // 1_000_000_000)

def add_forward_returns(df, price_cache, horizons):
    df = df.copy()
    for h in horizons:
        col = f"d_px_{h}"
        out = np.full(len(df), np.nan)
        for i, r in df.iterrows():
            key = (r["meeting_id"], r["sym"])
            series = price_cache.get(key)
            if series is None or pd.isna(r["timestamp_utc"]):
                continue
            t0 = sec_of(r["timestamp_utc"])
            try:
                p0 = series.loc[:t0].iloc[-1]
            except Exception:
                continue
            try:
                p1 = series.loc[:t0 + h].iloc[-1]
            except Exception:
                p1 = p0
            out[i] = p1 - p0
        df[col] = out
    return df

def winsorize(s: pd.Series, sigma: float) -> pd.Series:
    s = s.astype("float64")
    std = s.std(skipna=True, ddof=0)
    cap = sigma * std
    return s.clip(-cap, cap)

# ── Regression per asset & horizon ─────────────────────────────────────────
def run_regressions(df, asset, horizons, winsor_sigma):
    out_rows = []
    dfa = df.loc[df["sym"] == asset].copy()
    dfa["emotion"] = dfa["emotion"].str.lower()
    dfa["emo_happy"]    = (dfa["emotion"] == "happy").astype(int)
    dfa["emo_surprise"] = (dfa["emotion"] == "surprise").astype(int)
    dfa["emo_anxious"]  = (dfa["emotion"] == "anxious").astype(int)

    # save counts for sanity
    cnt_path = os.path.join(OUT_RESULTS, f"counts_{asset}.csv")
    (dfa["emotion"].value_counts().rename("count")).to_csv(cnt_path)
    print(f"[write] {cnt_path}")

    for h in horizons:
        yraw = dfa[f"d_px_{h}"]
        dfa[f"y{h}"] = winsorize(yraw, winsor_sigma)
        formula = f"y{h} ~ emo_happy + emo_surprise + emo_anxious + pre_px_60"
        fit = smf.ols(formula, data=dfa).fit(
            cov_type="cluster", cov_kwds={"groups": dfa["meeting_id"]}
        )
        for term in fit.params.index:
            out_rows.append({
                "asset": asset,
                "horizon_s": h,
                "term": term,
                "coef": float(fit.params[term]),
                "se":   float(fit.bse[term]),
                "pval": float(fit.pvalues[term]),
                "nobs": int(fit.nobs),
                "r2":   float(fit.rsquared),
            })
    return pd.DataFrame(out_rows)

def plot_emotion_paths(tidy, asset, outfile):
    emo_terms = ["emo_happy", "emo_surprise", "emo_anxious"]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for term, marker in zip(emo_terms, ["o", "s", "D"]):
        sub = tidy[(tidy["asset"] == asset) & (tidy["term"] == term)].sort_values("horizon_s")
        if sub.empty: 
            continue
        ax.errorbar(sub["horizon_s"], sub["coef"], yerr=1.96*sub["se"],
                    fmt=f"{marker}-", capsize=2, label=term.replace("emo_", "").title())
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Coefficient (ΔPrice)")
    ax.set_title(f"{asset}: Emotion coefficients by horizon (neutral baseline)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outfile, bbox_inches="tight")
    plt.close(fig)
    print(f"[write] {outfile}")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("[info] reading segments:", SEGMENTS_URI)
    seg = read_parquet(SEGMENTS_URI)[["meeting_id","segment_id","emotion"]]
    seg["emotion"] = seg["emotion"].str.lower()
    seg = seg.drop_duplicates(subset=["meeting_id","segment_id"], keep="last")

    print("[info] reading returns :", RETURNS_URI)
    rets = read_parquet(RETURNS_URI)

    # minimal required columns (do NOT require video_id)
    required = {"meeting_id","segment_id","sym","timestamp_utc","pre_px_60"}
    missing = required - set(rets.columns)
    if missing:
        sys.exit(f"[error] returns missing columns: {missing}")

    # attach/fill emotion if needed
    if "emotion" not in rets.columns or rets["emotion"].isna().any():
        rets = rets.merge(
            seg, on=["meeting_id","segment_id"], how="left", validate="many_to_one"
        )

    # build price cache once
    mtg_ids = rets["meeting_id"].dropna().unique().tolist()
    print(f"[info] building price cache for {len(mtg_ids)} meetings × {len(ASSETS)} assets")
    cache = build_price_cache(mtg_ids)

    # compute forward returns
    print(f"[info] computing forward returns for horizons: {HORIZONS}")
    df = add_forward_returns(rets, cache, HORIZONS)

    # regress per asset
    all_tidy = []
    for asset in ASSETS:
        print(f"\n=== {asset}: regressions across horizons ===")
        tidy = run_regressions(df, asset, HORIZONS, WINSOR_SIGMA)
        all_tidy.append(tidy)
    tidy_all = pd.concat(all_tidy, ignore_index=True)

    # write tidy CSV and plots
    tidy_path = os.path.join(OUT_RESULTS, "per_emotion_horizon_tidy.csv")
    tidy_all.to_csv(tidy_path, index=False)
    print(f"[write] {tidy_path}")

    for asset in ASSETS:
        out_png = os.path.join(OUT_FIGS, f"{asset}_emotion_coef_by_horizon.png")
        plot_emotion_paths(tidy_all, asset, out_png)

    with open(os.path.join(OUT_RESULTS, "horizon_diagnostics.txt"), "w") as f:
        f.write(str({
            "assets": ASSETS,
            "n_obs": int(len(df)),
            "meetings": int(df["meeting_id"].nunique()),
            "winsor_sigma": WINSOR_SIGMA,
            "horizons": HORIZONS
        }))

    print("\n[done] Horizon sweep complete.")
    print(f"Results → {OUT_RESULTS}")
    print(f"Figures → {OUT_FIGS}")

if __name__ == "__main__":
    main()