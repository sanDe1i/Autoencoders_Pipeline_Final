#!/usr/bin/env python3
"""Stage-5 grouped CV: random vs PDB-entry vs gene holdout.

Per-fold column-mean imputation is fit on the training rows only (same
hygiene as q6_train_dm_ae.py train-only μ/σ). Quote the *gene* row if the
claim is new-kinase prediction of z; the random row is interpolation.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from lightgbm import LGBMRegressor

# import distance-matrix builder from sibling Stage-5 script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict_v9_lgbm_shap import build_distance_matrix, impute_from_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conserved-csv", required=True, type=Path)
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--full-pdb-dir", required=True, type=Path)
    ap.add_argument("--latent-csv", required=True, type=Path)
    ap.add_argument("--ape-resi-floor", type=int, default=9999)
    ap.add_argument("--min-pair-coverage", type=float, default=0.75)
    ap.add_argument("--max-imputed-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=25)
    ap.add_argument("--out-txt", type=Path, default=None)
    a = ap.parse_args()

    X, manifest, pairs, frac = build_distance_matrix(
        a.conserved_csv, a.manifest_csv, a.full_pdb_dir,
        "6UANC", a.ape_resi_floor, a.min_pair_coverage)
    lat = pd.read_csv(a.latent_csv, keep_default_na=False)
    lat["chain_key"] = lat["chain_key"].astype(str).str.upper()
    lat = lat.set_index("chain_key")
    keys = manifest["chain_key"].astype(str).str.upper().values
    keep = np.array([k in lat.index for k in keys]) & (frac <= a.max_imputed_frac)
    X = X[keep]
    keys = keys[keep]
    man = manifest[keep].reset_index(drop=True)
    Y = lat.loc[list(keys), ["z0", "z1"]].to_numpy(float)
    lines = [
        f"chains={len(X)} features={X.shape[1]}",
        "imputation: per-fold train-only column mean",
    ]
    print(lines[0], flush=True)
    print(lines[1], flush=True)

    groups = {
        "random": None,
        "PDB entry": np.array([k[:4] for k in keys]),
        "gene": man["gene"].astype(str).values,
    }
    for name, g in groups.items():
        cv = KFold(5, shuffle=True, random_state=a.seed) if g is None else GroupKFold(5)
        r2s = []
        splits = cv.split(X) if g is None else cv.split(X, groups=g)
        for tr, te in splits:
            Xtr, (Xte,), _ = impute_from_train(X[tr], [X[te]])
            pr = np.column_stack([
                LGBMRegressor(n_estimators=400, num_leaves=31, verbose=-1,
                              random_state=a.seed)
                .fit(Xtr, Y[tr, j]).predict(Xte) for j in range(2)])
            ss = ((Y[te] - pr) ** 2).sum()
            tot = ((Y[te] - Y[te].mean(0)) ** 2).sum()
            r2s.append(1 - ss / tot)
        r2s = np.array(r2s)
        line = (f"  {name:10s} R2 = {r2s.mean():.3f} +/- {r2s.std():.3f}   "
                f"folds={np.round(r2s, 3)}")
        print(line, flush=True)
        lines.append(line)
    if a.out_txt:
        a.out_txt.parent.mkdir(parents=True, exist_ok=True)
        a.out_txt.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
