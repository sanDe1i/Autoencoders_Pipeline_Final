"""Marco follow-up #6: SE(3)-invariant distance-matrix autoencoder.

Trains molearn's CNN2d_AE (encoder sees only the pairwise Ca-Ca distance matrix,
so the latent is rotation/translation-invariant by construction) on the 6,531
v9.1 activation loops (27 Ca each). Loss is on the DISTANCE MATRIX of the
reconstruction vs input (also SE(3)-invariant), so the whole pipeline is frame
independent -- unlike the coordinate FoldingNet, whose z0 was dominated by loop
placement (Marco Q1).

Model classes copied verbatim from molearn.models.CNN2d_AE (pure torch), so no
molearn install is needed -- runs on any torch>=2 CUDA env.

Outputs: q6_dm_latent.csv (idx, z0, z1) + q6_dm_ae.ckpt.

Splits (scientific, not the original 90/10-used-as-val scheme):
  train  — fit weights AND compute coordinate mean/std
  val    — checkpoint selection only (never test)
  test   — reported once after training; never used to pick the ckpt

``--split gene`` (default): val and test are held-out *genes*, so a
number you report as generalisation is about new kinases, not new
chains of kinases the model already saw.
``--split random``: chain-level holdout only — do not claim new-kinase
generalisation from that number.
"""
import argparse
import csv
import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn

# ----------------------- molearn CNN2d_AE (verbatim) -----------------------
class Encoder(nn.Module):
    def __init__(self, latent_dim, dims, channels):
        super().__init__()
        self.convs = nn.ModuleList()
        for in_ch, out_ch in zip(channels[:-1], channels[1:]):
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=True),
                nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.1, inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.finallayer = nn.Linear(channels[-1], latent_dim)
    def forward(self, x):
        for conv in self.convs: x = conv(x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.finallayer(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, dims, channels):
        super().__init__()
        self.from_latent = nn.Linear(latent_dim, channels[-1]*dims[-1])
        self.dims = dims; self.channels = channels
        dims_rev = dims[::-1]; ch_rev = channels[::-1]
        layers = []
        for i in range(len(dims_rev)-1):
            h_in, h_out = dims_rev[i], dims_rev[i+1]
            in_ch = ch_rev[i]; default_out = ch_rev[i+1]
            is_last = (i == len(dims_rev)-2)
            out_ch = 3 if is_last else default_out
            op_h = h_out - 2*h_in
            layers.append(nn.ConvTranspose1d(in_ch, out_ch, 4, 2, 1, op_h, bias=True))
            if not is_last:
                layers += [nn.BatchNorm1d(out_ch), nn.LeakyReLU(0.1, inplace=True)]
        self.convs = nn.Sequential(*layers)
    def forward(self, z):
        z = z.view(z.size(0), -1)
        h = self.from_latent(z).view(z.size(0), -1, self.dims[-1])
        return self.convs(h)

class AutoEncoder(nn.Module):
    def __init__(self, dm_dim, latent_dim=2, init_c=32, m=2, min_size=9):
        super().__init__()
        dims, channels = self._dc(dm_dim, init_c, m, min_size)
        print(f"dims={dims} channels={channels}")
        self.dims = dims; self.channels = channels
        self.encoder = Encoder(latent_dim, dims, channels)
        self.decoder = Decoder(latent_dim, dims, channels)
    def _dc(self, dm_dim, init_c, m, min_size):
        dims=[dm_dim]; channels=[1]; curr=dm_dim; ch=init_c
        while curr >= min_size:
            channels.append(ch); curr=(curr+2-4)//2+1; dims.append(curr); ch=int(ch*m)
        return dims, channels
    @staticmethod
    def coords_to_dm(coord):
        n = coord.size(1)
        G = torch.bmm(coord, coord.transpose(1, 2))
        Gt = torch.diagonal(G, dim1=-2, dim2=-1)[:, None, :].repeat(1, n, 1)
        dm = torch.clamp(Gt + Gt.transpose(1, 2) - 2*G, min=1e-12)
        return torch.sqrt(dm)[:, None, :, :]
    def encode(self, x): return self.encoder(self.coords_to_dm(x))
    def decode(self, z): return self.decoder(z).squeeze(-1).permute(0, 2, 1)
    def forward(self, x): return self.decode(self.encode(x))

# ----------------------- data + training -----------------------
def load_pdb(path):
    out, cur = [], []
    for line in open(path):
        if line.startswith("MODEL"): cur = []
        elif line.startswith("ATOM") and line[12:16].strip() == "CA":
            cur.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        elif line.startswith("ENDMDL") and cur: out.append(cur); cur = []
    return np.asarray(out, np.float32)


def _assign_genes_by_chain_frac(genes, rng, test_frac, val_frac):
    """Hold out genes until ~test_frac / ~val_frac of *chains* are assigned.

    Genes are shuffled then greedily filled so large kinases (EGFR, ABL1)
    cannot all land in train by accident of '10% of gene names'.
    """
    order = rng.permutation(np.unique(genes))
    n = len(genes)
    test_genes, val_genes = [], []
    n_test = n_val = 0
    for g in order:
        ng = int((genes == g).sum())
        if n_test < test_frac * n:
            test_genes.append(g)
            n_test += ng
        elif n_val < val_frac * n:
            val_genes.append(g)
            n_val += ng
    test_genes, val_genes = set(test_genes), set(val_genes)
    overlap = test_genes & val_genes
    if overlap:
        raise RuntimeError(f"gene split overlap: {overlap}")
    te = np.where(np.isin(genes, list(test_genes)))[0]
    va = np.where(np.isin(genes, list(val_genes)))[0]
    tr = np.where(~np.isin(genes, list(test_genes | val_genes)))[0]
    return tr, va, te, sorted(test_genes), sorted(val_genes)


def make_split(n, rng, split, genes, test_frac, val_frac):
    if split == "random":
        perm = rng.permutation(n)
        n_te = max(1, int(round(n * test_frac)))
        n_va = max(1, int(round(n * val_frac)))
        te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
        return tr, va, te, None, None
    if genes is None:
        raise SystemExit("--split gene requires --manifest-csv with a gene column")
    if len(genes) != n:
        raise SystemExit(f"manifest genes {len(genes)} != n_models {n}")
    return _assign_genes_by_chain_frac(genes, rng, test_frac, val_frac)


def dm_mse(net, x, mse):
    dm_in = AutoEncoder.coords_to_dm(x)
    dm_out = AutoEncoder.coords_to_dm(net.decode(net.encode(x)))
    return mse(dm_out, dm_in).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combined-pdb", required=True)
    ap.add_argument("--out-latent", required=True)
    ap.add_argument("--out-ckpt", required=True)
    ap.add_argument("--manifest-csv", default=None,
                    help="Required for --split gene (must contain gene).")
    ap.add_argument("--split", choices=("gene", "random"), default="gene",
                    help="gene = hold out kinases (claim new-kinase gen.); "
                         "random = chain holdout only.")
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=25)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", dev, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

    X = load_pdb(a.combined_pdb)
    n, natoms, _ = X.shape
    print(f"loops {n} x {natoms} Ca")

    genes = None
    if a.manifest_csv:
        import pandas as pd
        man = pd.read_csv(a.manifest_csv, keep_default_na=False)
        genes = man["gene"].fillna("").astype(str).to_numpy()

    tr, va, te, test_genes, val_genes = make_split(
        n, rng, a.split, genes, a.test_frac, a.val_frac)
    print(f"split={a.split}  train {len(tr)}  val {len(va)}  test {len(te)}")
    if test_genes is not None:
        print(f"  val genes ({len(val_genes)}): {val_genes[:12]}{'...' if len(val_genes)>12 else ''}")
        print(f"  test genes ({len(test_genes)}): {test_genes[:12]}{'...' if len(test_genes)>12 else ''}")
        if genes is not None:
            print(f"  gene overlap train∩val∩test must be empty: "
                  f"{len(set(genes[tr]) & set(genes[va]) & set(genes[te]))} "
                  f"(train∩val={len(set(genes[tr])&set(genes[va]))}, "
                  f"train∩test={len(set(genes[tr])&set(genes[te]))}, "
                  f"val∩test={len(set(genes[va])&set(genes[te]))})")

    # mean/std from TRAIN coordinates only, then applied to val/test/full.
    mu = float(X[tr].mean())
    sd = float(X[tr].std())
    if sd < 1e-8:
        raise SystemExit("train coordinate std is ~0")
    Xs = torch.tensor((X - mu) / sd, dtype=torch.float32)
    print(f"train-only mean {mu:.4f}  std {sd:.4f}")

    Xtr = Xs[tr].to(dev)
    Xva = Xs[va].to(dev)
    Xte = Xs[te].to(dev)

    net = AutoEncoder(dm_dim=natoms, latent_dim=2).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    best_val = float("inf")
    best_state = None
    best_ep = -1
    for ep in range(a.epochs):
        net.train()
        idx = torch.randperm(len(Xtr))
        tot = 0.0
        for i in range(0, len(Xtr), a.batch):
            b = Xtr[idx[i:i + a.batch]]
            dm_in = AutoEncoder.coords_to_dm(b)
            dm_out = AutoEncoder.coords_to_dm(net.decode(net.encode(b)))
            loss = mse(dm_out, dm_in)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        # checkpoint on VAL only — never on test
        net.eval()
        with torch.no_grad():
            vloss = dm_mse(net, Xva, mse)
        if vloss < best_val:
            best_val = vloss
            best_state = copy.deepcopy(net.state_dict())
            best_ep = ep
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"epoch {ep:4d}  train {tot/len(Xtr):.5f}  "
                  f"val_dm_mse {vloss:.5f}  best {best_val:.5f} @ {best_ep}",
                  flush=True)

    if best_state is None:
        raise RuntimeError("no checkpoint selected")
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        test_loss = dm_mse(net, Xte, mse)
    claim = ("NEW-KINASE holdout" if a.split == "gene"
             else "RANDOM-CHAIN holdout (not new-kinase gen.)")
    print(f"BEST_VAL_DM_MSE {best_val:.5f}  epoch {best_ep}", flush=True)
    print(f"TEST_DM_MSE {test_loss:.5f}  [{claim}]", flush=True)

    Z = []
    with torch.no_grad():
        for i in range(0, n, 256):
            Z.append(net.encode(Xs[i:i + 256].to(dev)).cpu().numpy())
    Z = np.concatenate(Z, 0)
    out_lat = Path(a.out_latent)
    out_lat.parent.mkdir(parents=True, exist_ok=True)
    with out_lat.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "z0", "z1"])
        for i in range(n):
            w.writerow([i, Z[i, 0], Z[i, 1]])

    ckpt_path = Path(a.out_ckpt)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    split_dir = ckpt_path.parent
    np.savetxt(split_dir / "train_idx.txt", tr, fmt="%d")
    np.savetxt(split_dir / "val_idx.txt", va, fmt="%d")
    np.savetxt(split_dir / "test_idx.txt", te, fmt="%d")
    torch.save({
        "model_state_dict": net.state_dict(),
        "mu": mu,
        "sd": sd,
        "split": a.split,
        "seed": a.seed,
        "best_epoch": best_ep,
        "best_val_dm_mse": best_val,
        "test_dm_mse": test_loss,
        "train_idx": tr,
        "val_idx": va,
        "test_idx": te,
        "test_genes": test_genes,
        "val_genes": val_genes,
    }, ckpt_path)
    print(f"wrote {out_lat}  z0 std {Z[:,0].std():.3f}  z1 std {Z[:,1].std():.3f}")
    print(f"wrote {ckpt_path}")


if __name__ == "__main__":
    main()
