"""Plot the Stage-4 2-D latent coloured by kinase gene and by DFG state.

Existing Stage-7 scripts overlay *drugs* (or ligand types) on the latent.
This helper draws the full 6,531-chain landscape grouped by kinase (gene).

Because there are ~276 genes, only the N most frequent kinases get unique
colours; the rest are drawn as a grey 'other' cloud so the figure stays
readable.

Outputs (under --out-dir):
  kinase_latent_by_gene.{png,pdf}
      one scatter; colour = kinase (top-N genes + other)
  kinase_latent_by_gene_dfg.{png,pdf}
      same colouring; marker = DFG-in / DFG-out / DFG-inter
  kinase_latent_by_dfg.{png,pdf}
      all points coloured + marked by DFG state only
  kinase_latent_top_genes.csv
      chain counts for the highlighted genes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# Distinct colours for the highlighted kinases (tab20, skip greys).
GENE_PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b",
    "#17becf", "#bcbd22", "#7f7f7f", "#aec7e8", "#ffbb78",
]

DFG_STYLE = {
    "DFGin":    {"marker": "o", "color": "#4c78a8", "label": "DFG-in"},
    "DFGout":   {"marker": "^", "color": "#f58518", "label": "DFG-out"},
    "DFGinter": {"marker": "s", "color": "#54a24b", "label": "DFG-inter"},
}
DFG_OTHER = {"marker": "x", "color": "#bbbbbb", "label": "DFG unknown"}
OTHER_COLOR = "#d9d9d9"


def _norm_dfg(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({
        "": "unknown",
        "nan": "unknown",
        "None": "unknown",
        "NaN": "unknown",
        "<NA>": "unknown",
    })
    return s


def _save(fig: plt.Figure, out: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out / stem}.png")


def plot_by_gene(df: pd.DataFrame, top_genes: list[str],
                 colors: dict[str, str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    other = ~df["gene"].isin(top_genes)
    ax.scatter(df.loc[other, "z0"], df.loc[other, "z1"],
               s=8, c=OTHER_COLOR, alpha=0.35, linewidths=0,
               rasterized=True, zorder=1, label=f"other (n={int(other.sum())})")
    for gene in reversed(top_genes):  # draw rarest of the top-N last
        m = df["gene"] == gene
        ax.scatter(df.loc[m, "z0"], df.loc[m, "z1"],
                   s=16, c=colors[gene], alpha=0.75, linewidths=0,
                   rasterized=True, zorder=2,
                   label=f"{gene} (n={int(m.sum())})")
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title("Activation-loop latent, grouped by kinase gene\n"
                 f"top {len(top_genes)} genes coloured; remaining genes in grey")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8, markerscale=1.4)
    fig.tight_layout()
    _save(fig, out, "kinase_latent_by_gene")


def plot_by_gene_and_dfg(df: pd.DataFrame, top_genes: list[str],
                         colors: dict[str, str], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    other = ~df["gene"].isin(top_genes)
    ax.scatter(df.loc[other, "z0"], df.loc[other, "z1"],
               s=7, c=OTHER_COLOR, alpha=0.30, linewidths=0, marker="o",
               rasterized=True, zorder=1)

    for gene in reversed(top_genes):
        sub = df[df["gene"] == gene]
        for dfg, st in DFG_STYLE.items():
            m = sub["dfg"] == dfg
            if not m.any():
                continue
            ax.scatter(sub.loc[m, "z0"], sub.loc[m, "z1"],
                       s=22, c=colors[gene], marker=st["marker"],
                       alpha=0.85, linewidths=0.2, edgecolors="white",
                       rasterized=True, zorder=3)
        unk = ~sub["dfg"].isin(DFG_STYLE)
        if unk.any():
            ax.scatter(sub.loc[unk, "z0"], sub.loc[unk, "z1"],
                       s=16, c=colors[gene], marker="x",
                       alpha=0.7, linewidths=0.8, rasterized=True, zorder=2)

    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title("Latent by kinase (colour) and DFG state (marker)")

    gene_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=colors[g], markeredgecolor="none",
               markersize=8, label=f"{g} (n={int((df['gene']==g).sum())})")
        for g in top_genes
    ]
    gene_handles.append(
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=OTHER_COLOR, markeredgecolor="none",
               markersize=8, label=f"other (n={int(other.sum())})")
    )
    dfg_handles = [
        Line2D([0], [0], marker=st["marker"], color="0.2",
               markerfacecolor="0.2", markeredgecolor="white",
               markersize=8, linestyle="none", label=st["label"])
        for st in DFG_STYLE.values()
    ]
    dfg_handles.append(
        Line2D([0], [0], marker="x", color="0.4",
               markersize=7, linestyle="none", label=DFG_OTHER["label"])
    )
    leg1 = ax.legend(handles=gene_handles, title="Kinase",
                     loc="upper left", bbox_to_anchor=(1.02, 1.0),
                     frameon=False, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=dfg_handles, title="DFG",
              loc="upper left", bbox_to_anchor=(1.02, 0.42),
              frameon=False, fontsize=8)
    fig.tight_layout()
    _save(fig, out, "kinase_latent_by_gene_dfg")


def plot_by_dfg(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    order = list(DFG_STYLE.keys()) + ["unknown"]
    for dfg in reversed(order):
        if dfg == "unknown":
            m = ~df["dfg"].isin(DFG_STYLE)
            st = DFG_OTHER
        else:
            m = df["dfg"] == dfg
            st = DFG_STYLE[dfg]
        if not m.any():
            continue
        ax.scatter(df.loc[m, "z0"], df.loc[m, "z1"],
                   s=18 if dfg != "unknown" else 12,
                   c=st["color"], marker=st["marker"],
                   alpha=0.65 if dfg != "unknown" else 0.45,
                   linewidths=0.15, edgecolors="white",
                   rasterized=True,
                   label=f"{st['label']} (n={int(m.sum())})")
    ax.set_xlabel("z0")
    ax.set_ylabel("z1")
    ax.set_title("Activation-loop latent coloured by Kincore DFG state")
    ax.legend(loc="best", frameon=False, fontsize=9, markerscale=1.3)
    fig.tight_layout()
    _save(fig, out, "kinase_latent_by_dfg")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n-top-genes", type=int, default=12,
                    help="How many most frequent kinase genes to colour.")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.latent_csv, keep_default_na=False)
    if not {"z0", "z1", "gene"}.issubset(df.columns):
        raise SystemExit("latent CSV must have columns z0, z1, gene")
    df["gene"] = df["gene"].astype(str).str.strip().replace({"": "UNKNOWN"})
    dfg_col = "dfg_spatial" if "dfg_spatial" in df.columns else None
    if dfg_col is None:
        raise SystemExit("latent CSV must have dfg_spatial")
    df["dfg"] = _norm_dfg(df[dfg_col])

    counts = df["gene"].value_counts()
    n_top = min(args.n_top_genes, len(counts))
    top_genes = counts.index[:n_top].tolist()
    colors = {g: GENE_PALETTE[i % len(GENE_PALETTE)]
              for i, g in enumerate(top_genes)}
    print(f"chains={len(df)}  genes={df['gene'].nunique()}  "
          f"highlighting top {n_top}")
    print(counts.head(n_top).to_string())
    print("DFG counts:\n", df["dfg"].value_counts(dropna=False).to_string())

    pd.DataFrame({
        "gene": top_genes,
        "n_chains": [int(counts[g]) for g in top_genes],
        "rank": list(range(1, n_top + 1)),
        "color": [colors[g] for g in top_genes],
    }).to_csv(args.out_dir / "kinase_latent_top_genes.csv", index=False)

    plot_by_gene(df, top_genes, colors, args.out_dir)
    plot_by_gene_and_dfg(df, top_genes, colors, args.out_dir)
    plot_by_dfg(df, args.out_dir)
    print(f"Done. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
