"""Embed chains EXACTLY as molearn trained them — using PDBData's own
standardisation (global (coords-mean)/std, no per-chain centering),
instead of embed_v9_ca_only.py's raw per-chain centering.

This eliminates the train/inference normalisation mismatch: the encoder
is fed the same input distribution it was trained on. Writes a latent
CSV in the same schema as embed_v9_ca_only.py so the downstream is
unchanged.
"""
from __future__ import annotations

import argparse
import copy as _copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-pdb", required=True, type=Path)
    ap.add_argument("--manifest-csv", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--out-csv", required=True, type=Path)
    args = ap.parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    from molearn.data import PDBData
    from molearn.models.small_foldingnet import Small_AutoEncoder

    # IDENTICAL loading path to train_v9_ca_only / train_v91_regularized.
    data = PDBData()
    data.import_pdb(filename=str(args.combined_pdb))
    data.fix_terminal()
    data.atomselect(atoms=["CA"])
    data.prepare_dataset()                # computes + applies global mean/std
    print(f"molearn standardisation: mean={data.mean:.4f}, std={data.std:.4f}")
    X = data.dataset                      # (n, n_atoms, 3), standardised
    n, n_atoms, _ = X.shape
    print(f"dataset: {X.shape}")

    manifest = pd.read_csv(args.manifest_csv, keep_default_na=False)
    if len(manifest) != n:
        raise SystemExit(f"manifest {len(manifest)} != models {n}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Small_AutoEncoder(out_points=n_atoms).to(device)
    state = torch.load(str(args.checkpoint), map_location=device,
                       weights_only=False)
    sd = state.get("model_state_dict", state.get("state_dict", state))
    net.load_state_dict(sd)
    net.eval()

    Xd = X.to(device)
    Z = []
    with torch.no_grad():
        for i in range(0, n, 128):
            z = net.encode(Xd[i:i + 128].float())
            Z.append(z.cpu().numpy().reshape(z.shape[0], -1))
    Z = np.concatenate(Z, axis=0)
    print(f"latent: z0 std={Z[:,0].std():.3f} z1 std={Z[:,1].std():.3f} "
          f"|z|max={np.hypot(Z[:,0], Z[:,1]).max():.2f}")

    df = manifest.copy()
    df["idx"] = np.arange(n)
    df["z0"] = Z[:, 0]
    df["z1"] = Z[:, 1] if Z.shape[1] > 1 else 0.0
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
