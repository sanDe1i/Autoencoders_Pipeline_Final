"""SE3 latent diagnostics: train/val/test marginals + DFG scatter.

Supports the three-way gene (or random) split from q6_train_dm_ae.py.
Do not use the FoldingNet plotter that requires train+test to cover every row.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DFG_COLORS = {
    "DFGin": "#4c78a8",
    "DFGout": "#f58518",
    "DFGinter": "#54a24b",
}
SPLIT_COLORS = {
    "train": "#4c78a8",
    "val": "#54a24b",
    "test": "#f58518",
}


def load_indices(path: Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path, dtype=int))


def assign_split(df: pd.DataFrame, parts: dict[str, np.ndarray]) -> pd.DataFrame:
    out = df.copy()
    out["ae_split"] = ""
    seen: set[int] = set()
    for name, ids in parts.items():
        if ids.min() < 0 or ids.max() >= len(out):
            raise SystemExit(f"{name} idx out of range")
        hit = seen.intersection(ids.tolist())
        if hit:
            raise SystemExit(f"split overlap in {name}")
        seen.update(ids.tolist())
        out.loc[ids, "ae_split"] = name
    missing = int((out["ae_split"] == "").sum())
    if missing:
        raise SystemExit(f"indices miss {missing} rows")
    return out


def plot_split_hist(df: pd.DataFrame, out_dir: Path) -> None:
    splits = [s for s in ("train", "val", "test") if (df["ae_split"] == s).any()]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, col in zip(axes, ("z0", "z1")):
        for name in splits:
            sub = df[df["ae_split"] == name]
            ax.hist(
                sub[col],
                bins=50,
                alpha=0.5,
                density=True,
                label=f"{name} (n={len(sub)})",
                color=SPLIT_COLORS[name],
            )
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.set_title(f"{col} by AE split")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SE3 latent marginals (test = held-out kinases if --split gene)", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"latent_z0_z1_split_distribution.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dfg_hist(df: pd.DataFrame, dfg_col: str, out_dir: Path) -> None:
    labeled = df[dfg_col].fillna("None").astype(str).replace("", "None")
    order = [k for k in DFG_COLORS if (labeled == k).any()]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for ax, col in zip(axes, ("z0", "z1")):
        for name in order:
            sub = df[labeled == name]
            ax.hist(
                sub[col],
                bins=50,
                alpha=0.5,
                density=True,
                label=f"{name} (n={len(sub)})",
                color=DFG_COLORS[name],
            )
        ax.set_xlabel(col)
        ax.set_ylabel("density")
        ax.set_title(f"{col} by DFG-spatial")
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SE3 latent marginals by DFG-spatial", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"latent_z0_z1_dfg_distribution.{ext}",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_by_dfg(df: pd.DataFrame, dfg_col: str, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    labeled = df[dfg_col].fillna("None").astype(str)
    for label, color in DFG_COLORS.items():
        sub = df[labeled == label]
        if sub.empty:
            continue
        ax.scatter(sub["z0"], sub["z1"], s=8, alpha=0.72, linewidths=0,
                   c=color, label=f"{label} (n={len(sub)})")
    unlabeled = df[~labeled.isin(DFG_COLORS)]
    if not unlabeled.empty:
        ax.scatter(unlabeled["z0"], unlabeled["z1"], s=6, alpha=0.35,
                   linewidths=0, c="#9d9d9d", label=f"unlabeled (n={len(unlabeled)})")
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title("SE3 latent colored by DFG-spatial")
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"latent_scatter_by_dfg.{ext}", dpi=300)
    plt.close(fig)


def plot_by_split(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    for name, color in SPLIT_COLORS.items():
        sub = df[df["ae_split"] == name]
        if sub.empty:
            continue
        ax.scatter(sub["z0"], sub["z1"], s=8, alpha=0.65, linewidths=0,
                   c=color, label=f"{name} (n={len(sub)})")
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title("SE3 latent colored by train / val / test")
    ax.legend(frameon=False, fontsize=8, markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"latent_scatter_by_split.{ext}", dpi=300)
    plt.close(fig)


def write_summary(df: pd.DataFrame, dfg_col: str, out_dir: Path) -> None:
    rows = []
    splits = ["train", "val", "test", "all"]
    for split in splits:
        sub = df if split == "all" else df[df["ae_split"] == split]
        if sub.empty and split != "all":
            continue
        row = {
            "split": split,
            "n": len(sub),
            "z0_mean": float(sub["z0"].mean()),
            "z0_std": float(sub["z0"].std()),
            "z1_mean": float(sub["z1"].mean()),
            "z1_std": float(sub["z1"].std()),
        }
        for label in (*DFG_COLORS,):
            row[f"n_{label}"] = int((sub[dfg_col].astype(str) == label).sum())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "latent_plot_summary.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--train-idx", type=Path, default=None)
    ap.add_argument("--val-idx", type=Path, default=None)
    ap.add_argument("--test-idx", type=Path, default=None)
    ap.add_argument("--dfg-col", default="dfg_spatial")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.latent_csv, keep_default_na=False)
    for col in ("z0", "z1"):
        if col not in df.columns:
            raise SystemExit(f"missing {col}")

    if "ae_split" in df.columns and df["ae_split"].astype(str).ne("").any():
        pass
    elif all(p is not None for p in (args.train_idx, args.val_idx, args.test_idx)):
        df = assign_split(df, {
            "train": load_indices(args.train_idx),
            "val": load_indices(args.val_idx),
            "test": load_indices(args.test_idx),
        })
    else:
        raise SystemExit("need ae_split column or --train-idx --val-idx --test-idx")

    plot_split_hist(df, args.out_dir)
    plot_dfg_hist(df, args.dfg_col, args.out_dir)
    plot_by_dfg(df, args.dfg_col, args.out_dir)
    plot_by_split(df, args.out_dir)
    write_summary(df, args.dfg_col, args.out_dir)
    print(f"wrote figures under {args.out_dir}")
    print(df["ae_split"].value_counts().to_string())
    print(df[args.dfg_col].replace("", "None").value_counts().head(8).to_string())


if __name__ == "__main__":
    main()
