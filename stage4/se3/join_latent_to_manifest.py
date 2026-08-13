#!/usr/bin/env python3
"""Join Stage-3 SE3 latent (idx, z0, z1) onto Stage-1 manifest.

Stage-3 ``q6_train_dm_ae.py`` already encodes all models; this does NOT
reload FoldingNet / molearn. ``embed_molearn_norm.py`` is incompatible
with the SE3 checkpoint.

Optionally attaches ``ae_split`` from the gene/random train/val/test
index files written next to the ckpt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_idx(path: Path) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(path, dtype=int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-idx-csv", required=True, type=Path)
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    ap.add_argument("--train-idx", type=Path, default=None)
    ap.add_argument("--val-idx", type=Path, default=None)
    ap.add_argument("--test-idx", type=Path, default=None)
    a = ap.parse_args()

    lat = pd.read_csv(a.latent_idx_csv)
    man = pd.read_csv(a.manifest_csv, keep_default_na=False)
    for col in ("idx", "z0", "z1"):
        if col not in lat.columns:
            raise SystemExit(f"latent CSV missing {col!r} — not SE3 idx output")
    if len(lat) != len(man):
        raise SystemExit(f"length mismatch latent={len(lat)} manifest={len(man)}")

    lat = lat.sort_values("idx").reset_index(drop=True)
    idx = lat["idx"].to_numpy(int)
    if not np.array_equal(idx, np.arange(len(lat))):
        raise SystemExit("latent idx is not 0..n-1 in MODEL order")
    if "model_idx" in man.columns:
        midx = pd.to_numeric(man["model_idx"], errors="coerce").to_numpy()
        if np.isfinite(midx).all() and not np.array_equal(midx.astype(int), idx):
            raise SystemExit("manifest model_idx does not match latent idx order")

    out = man.copy()
    out["z0"] = lat["z0"].astype(float).to_numpy()
    out["z1"] = lat["z1"].astype(float).to_numpy()

    idx_files = {
        "train": a.train_idx,
        "val": a.val_idx,
        "test": a.test_idx,
    }
    if any(p is not None for p in idx_files.values()):
        if not all(p is not None for p in idx_files.values()):
            raise SystemExit("pass all of --train-idx --val-idx --test-idx or none")
        split = np.array([""] * len(out), dtype=object)
        seen = set()
        for name, path in idx_files.items():
            ids = _load_idx(path)
            if ids.min() < 0 or ids.max() >= len(out):
                raise SystemExit(f"{name} idx out of range")
            overlap = seen.intersection(ids.tolist())
            if overlap:
                raise SystemExit(f"split idx overlap involving {name}")
            seen.update(ids.tolist())
            split[ids] = name
        if (split == "").any():
            raise SystemExit(
                f"split indices miss {int((split == '').sum())} rows"
            )
        out["ae_split"] = split
        print(
            "ae_split",
            {k: int((out.ae_split == k).sum()) for k in ("train", "val", "test")},
        )

    prefer = [
        "chain_key", "pdb", "chain", "gene", "group", "dfg_spatial",
        "dihedral", "ligand_type", "ae_split", "z0", "z1",
    ]
    cols = [c for c in prefer if c in out.columns] + [
        c for c in out.columns if c not in prefer
    ]
    a.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(a.out_csv, index=False)
    print(
        f"wrote {a.out_csv} rows={len(out)} "
        f"z0=[{out.z0.min():.3f},{out.z0.max():.3f}] "
        f"z1=[{out.z1.min():.3f},{out.z1.max():.3f}] "
        f"z0_std={out.z0.std():.3f} z1_std={out.z1.std():.3f}"
    )


if __name__ == "__main__":
    main()
