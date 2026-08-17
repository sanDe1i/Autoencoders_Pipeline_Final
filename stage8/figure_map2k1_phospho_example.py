"""Focused figure for the MAP2K1 phospho-site worked example.

Two panels:
  A. Global SE3 latent (all chains) coloured by Kincore DFG-spatial,
     with MAP2K1 WT / S218A+S222A / S218D+S222D overlaid.
  B. MAP2K1 zoom: WT centroid → phospho-dead / phospho-mimetic,
     with |Δ|, Mahalanobis distance (vs WT covariance), and a
     permutation p-value recomputed on *this* latent (never hardcode
     FoldingNet-era numbers — SE3 units are ~0–1).

Reads Stage-4 join CSV (chain_key, gene, dfg_spatial, z0, z1).
Writes <out-prefix>.{png,pdf} and <out-prefix>_stats.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from numpy.linalg import pinv


S218A_S222A_KEYS = (
    "6NYBB;6PP9B;6Q0JC;6Q0JD;6Q0TC;6V2WB;7M0TB;7M0UB;7M0VB;"
    "7M0WB;7M0XB;7M0YB;7M0ZB;8CHFE;8CHFF;8DGSB;8DGTB"
).split(";")
S218D_S222D_KEYS = ["5YT3B", "5YT3D"]

DFG_COLORS = {
    "DFGin": "#9CB7E0",
    "DFGinter": "#E8C26F",
    "DFGout": "#D67A7A",
    "": "#D0D0D0",
    "noise": "#D0D0D0",
}


def style():
    rcParams.update({
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "savefig.dpi": 600,
    })


def _fmt_p(p: float) -> str:
    if p <= 0:
        return "<1e-4"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.3f}"


def mutant_stats(wt_xy: np.ndarray, mut_xy: np.ndarray, n_perm: int, seed: int):
    """Euclidean |Δ|, Mahalanobis of mutant centroid vs WT cloud, perm p."""
    if len(wt_xy) < 3 or len(mut_xy) == 0:
        return dict(delta=np.nan, mahal=np.nan, perm_p=np.nan,
                    n_wt=len(wt_xy), n_mut=len(mut_xy))
    wt_c = wt_xy.mean(axis=0)
    mut_c = mut_xy.mean(axis=0)
    d = mut_c - wt_c
    delta = float(np.hypot(d[0], d[1]))
    cov = np.cov(wt_xy.T)
    if np.linalg.matrix_rank(cov) < 2:
        mahal = np.nan
    else:
        mahal = float(np.sqrt(d @ pinv(cov) @ d))

    # Permutation: draw n_mut chains from the pooled MAP2K1 set (WT+mut),
    # recompute centroid-|Δ| to the remaining WT; one-sided >= observed.
    pool = np.vstack([wt_xy, mut_xy])
    n_mut = len(mut_xy)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        idx = rng.choice(len(pool), size=n_mut, replace=False)
        mask = np.ones(len(pool), dtype=bool)
        mask[idx] = False
        if mask.sum() < 2:
            continue
        d_null = float(np.hypot(*(pool[idx].mean(0) - pool[mask].mean(0))))
        if d_null >= delta - 1e-12:
            ge += 1
    perm_p = (ge + 1) / (n_perm + 1)
    return dict(delta=delta, mahal=mahal, perm_p=perm_p,
                n_wt=len(wt_xy), n_mut=n_mut,
                mut_z0=float(mut_c[0]), mut_z1=float(mut_c[1]),
                wt_z0=float(wt_c[0]), wt_z1=float(wt_c[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--out-prefix", required=True, type=Path)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=25)
    args = ap.parse_args()

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    style()

    df = pd.read_csv(args.latent_csv, keep_default_na=False)
    print(f"Loaded {len(df)} chains")

    mek = df[df["gene"].astype(str).str.upper() == "MAP2K1"].copy()
    if mek.empty:
        raise SystemExit("No MAP2K1 chains in latent CSV")

    keys = set(mek["chain_key"].astype(str).str.upper())
    miss_a = [k for k in S218A_S222A_KEYS if k not in keys]
    miss_d = [k for k in S218D_S222D_KEYS if k not in keys]
    if miss_a or miss_d:
        raise SystemExit(f"Missing MAP2K1 mutant keys: dead={miss_a} mimetic={miss_d}")

    mek["mut_label"] = "WT / other"
    mek.loc[mek["chain_key"].isin(S218A_S222A_KEYS),
            "mut_label"] = "S218A+S222A (phospho-dead)"
    mek.loc[mek["chain_key"].isin(S218D_S222D_KEYS),
            "mut_label"] = "S218D+S222D (phospho-mimetic)"
    print(f"MAP2K1 chains: {len(mek)}")
    print(mek["mut_label"].value_counts().to_string())

    wt = mek[mek["mut_label"] == "WT / other"]
    pd_ = mek[mek["mut_label"] == "S218A+S222A (phospho-dead)"]
    pm = mek[mek["mut_label"] == "S218D+S222D (phospho-mimetic)"]
    if len(pd_) == 0 or len(pm) == 0 or len(wt) == 0:
        raise SystemExit(
            f"Need WT, phospho-dead and phospho-mimetic; got "
            f"WT={len(wt)} dead={len(pd_)} mimetic={len(pm)}")

    wt_xy = wt[["z0", "z1"]].to_numpy(float)
    stats_dead = mutant_stats(wt_xy, pd_[["z0", "z1"]].to_numpy(float),
                              args.n_perm, args.seed)
    stats_mim = mutant_stats(wt_xy, pm[["z0", "z1"]].to_numpy(float),
                             args.n_perm, args.seed + 1)
    print("\nphospho-dead:", {k: stats_dead[k] for k in
                              ("delta", "mahal", "perm_p", "n_mut")})
    print("phospho-mimetic:", {k: stats_mim[k] for k in
                               ("delta", "mahal", "perm_p", "n_mut")})

    stats_df = pd.DataFrame([
        {"mutant": "S218A+S222A", **stats_dead},
        {"mutant": "S218D+S222D", **stats_mim},
    ])
    stats_path = Path(str(args.out_prefix) + "_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Wrote {stats_path}")

    centroids = mek.groupby("mut_label")[["z0", "z1"]].mean()
    print(); print(centroids)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.7),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})

    axA = axes[0]
    for spatial in ("DFGin", "DFGinter", "DFGout"):
        sub = df[df["dfg_spatial"] == spatial]
        axA.scatter(sub["z0"], sub["z1"], s=4, alpha=0.30,
                    color=DFG_COLORS.get(spatial, "#D0D0D0"),
                    linewidths=0, zorder=1,
                    label=f"{spatial} (n={len(sub)})")
    others = df[~df["dfg_spatial"].isin({"DFGin", "DFGinter", "DFGout"})]
    if len(others):
        axA.scatter(others["z0"], others["z1"], s=4, alpha=0.20,
                    color="#D0D0D0", linewidths=0, zorder=1,
                    label=f"unlabelled (n={len(others)})")

    axA.scatter(wt["z0"], wt["z1"], s=30, alpha=0.85,
                color="#4D4D4D", edgecolor="black", linewidth=0.4,
                zorder=3, label=f"MAP2K1 WT / other (n={len(wt)})")
    axA.scatter(pd_["z0"], pd_["z1"], s=70, alpha=0.95,
                color="#3C78D8", marker="o",
                edgecolor="black", linewidth=0.5, zorder=4,
                label=f"S218A+S222A (n={len(pd_)})")
    axA.scatter(pm["z0"], pm["z1"], s=140, alpha=0.95,
                color="#CC0000", marker="*",
                edgecolor="black", linewidth=0.6, zorder=5,
                label=f"S218D+S222D (n={len(pm)})")
    axA.set_xlabel("z0")
    axA.set_ylabel("z1")
    axA.set_title("A. MAP2K1 chains in the global autoencoder latent space",
                  loc="left")
    axA.legend(loc="lower right", fontsize=9, markerscale=1.4)

    axB = axes[1]
    axB.scatter(wt["z0"], wt["z1"], s=45, alpha=0.85,
                color="#4D4D4D", edgecolor="black", linewidth=0.4,
                zorder=3, label=f"MAP2K1 WT / other (n={len(wt)})")
    axB.scatter(pd_["z0"], pd_["z1"], s=80, alpha=0.95,
                color="#3C78D8", marker="o",
                edgecolor="black", linewidth=0.5, zorder=4,
                label=f"S218A+S222A (n={len(pd_)})")
    axB.scatter(pm["z0"], pm["z1"], s=180, alpha=0.95,
                color="#CC0000", marker="*",
                edgecolor="black", linewidth=0.6, zorder=5,
                label=f"S218D+S222D (n={len(pm)})")

    wt_c = centroids.loc["WT / other"]
    pd_c = centroids.loc["S218A+S222A (phospho-dead)"]
    pm_c = centroids.loc["S218D+S222D (phospho-mimetic)"]

    axB.scatter([wt_c["z0"]], [wt_c["z1"]], marker="X", s=200,
                color="#101010", edgecolor="white", linewidth=1.5,
                zorder=6)
    axB.annotate("WT centroid", (wt_c["z0"], wt_c["z1"]),
                 xytext=(8, 8), textcoords="offset points",
                 fontsize=10, color="#101010", zorder=7)

    axB.annotate(
        "", xy=(pd_c["z0"], pd_c["z1"]),
        xytext=(wt_c["z0"], wt_c["z1"]),
        arrowprops=dict(arrowstyle="->", color="#3C78D8",
                        lw=2.0, alpha=0.85), zorder=6,
    )
    axB.annotate(
        "", xy=(pm_c["z0"], pm_c["z1"]),
        xytext=(wt_c["z0"], wt_c["z1"]),
        arrowprops=dict(arrowstyle="->", color="#CC0000",
                        lw=2.0, alpha=0.85), zorder=6,
    )

    # Place labels in *data* coordinates with a small pad in latent units
    # (SE3 scale ~0.1–1; old FoldingNet offsets of ±5 broke the panel).
    span = max(float(mek["z0"].max() - mek["z0"].min()),
               float(mek["z1"].max() - mek["z1"].min()), 0.2)
    pad = 0.04 * span
    midA = ((wt_c["z0"] + pd_c["z0"]) / 2, (wt_c["z1"] + pd_c["z1"]) / 2)
    midB = ((wt_c["z0"] + pm_c["z0"]) / 2, (wt_c["z1"] + pm_c["z1"]) / 2)
    mah_a = stats_dead["mahal"]
    mah_b = stats_mim["mahal"]
    axB.text(midA[0] - pad, midA[1] + pad,
             f"|Δ| = {stats_dead['delta']:.3f}\n"
             f"Mahal σ = {mah_a:.2f}\n"
             f"perm p = {_fmt_p(stats_dead['perm_p'])}",
             fontsize=10, color="#3C78D8", ha="right", va="bottom")
    axB.text(midB[0] + pad, midB[1] + pad,
             f"|Δ| = {stats_mim['delta']:.3f}\n"
             f"Mahal σ = {mah_b:.2f}\n"
             f"perm p = {_fmt_p(stats_mim['perm_p'])}",
             fontsize=10, color="#CC0000", ha="left", va="bottom")

    axB.set_xlabel("z0")
    axB.set_ylabel("z1")
    axB.set_title("B. MAP2K1 wild-type/other-to-mutant displacement",
                  loc="left")
    axB.legend(loc="upper right", fontsize=9, markerscale=1.0)

    fig.tight_layout()
    out_png = str(args.out_prefix) + ".png"
    out_pdf = str(args.out_prefix) + ".pdf"
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"\nWrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
