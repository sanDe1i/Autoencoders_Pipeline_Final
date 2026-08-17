# Autoencoders Pipeline Final — SE(3) scientific path

Runnable code tree for the kinase activation-loop **SE(3) distance-matrix
autoencoder** on the canonical **6,531** chains (5,318 + 1,213).

This tree is **not** tied to the old WORKFLOW FoldingNet / mixed-centroid
convention. Stage3 trains `q6_train_dm_ae.py` with
**gene holdout**, **train-only μ/σ**, and **val-only checkpointing**.

---

## Environment

```bash
conda env create -f environment.yml
```

External binary (Stage 2 only): **FoldMason**.

Canonical model path does **not** need `molearn` (CNN2d_AE is inlined in
`stage3/se3/q6_train_dm_ae.py`). Install `molearn==2.0.4` only for the
legacy FoldingNet scripts.

---

## Layout

```
Autoencoders_Pipeline_Final/
├── README.md
├── environment.yml
├── input/
│   ├── stage1/   # manifest_v9, addendum, 6UAN_full.pdb
│   ├── stage2/   # FoldMason seed residues
│   ├── stage7/   # Kincore PK_labels_PDB.fasta
│   └── stage9/   # OncoKB static TSVs
└──stage1/ … stage12/
```

Typical exports when running:

```bash
export LATENT="$OUTPUT/stage4/se3/v91_SE3_latent_seed25.csv"
export CONSERVED="$OUTPUT/stage2/v8_braf_mapped_conserved_residues.csv"
export MANIFEST="$OUTPUT/stage1/manifest_v91.csv"
export PDBS="$OUTPUT/stage1/PDBs"
```

All outputs go under `$OUTPUT/stageN/`.

---

## Stages (SE3 path)

| Stage | What | Main entry |
|------|------|------------|
| **1** | Download PDBs; build 6,531 × 27 Cα loops (same frame) | `stage1/download_pdbs.py`, `build_v91_dataset.py` |
| **2** | FoldMason conserved non-loop map | `stage2/map_v8_conserved_by_foldmason_chunks.py` |
| **3** | SE3 DM-AE, gene split, train-only μ/σ | `stage3/se3/q6_train_dm_ae.py` |
| **4** | Join latent ↔ manifest; plots | `stage4/se3/join_latent_to_manifest.py`, `plot_latent_distributions.py` |
| **5** | Conserved-distance → z (LightGBM + SHAP); optional grouped CV | `stage5/predict_v9_lgbm_shap.py`, `grouped_cv_cli.py` |
| **6** | FI robustness across methods | `stage6/eval_v9_fi_extended.py` (+ agreement / deeper / compare) |
| **7** | Drug / selectivity analyses | `stage7/v9_*.py` |
| **8** | MAP2K1 phospho example figure | `stage8/figure_map2k1_phospho_example.py` |
| **9** | Mutation collect → enumerate → significance → OncoKB | `stage9/*.py` |
| **10–12** | Extended analyses / MD project / variants | optional; deferred unless needed |

---

## Scientific conventions (read before quoting numbers)

1. **Latent** = SE3 2-D unit-scale `(z0, z1)` from Stage4 CSV. Not the old
   FoldingNet / mixed-centroid latent (`|z0|` up to hundreds of Å).
2. **Stage5 / Stage6 FI** default to **random chain split** so the model
   that produces SHAP actually predicts z (R² ≈ 0.9). Quote **gene
   holdout / gene 5-fold** (~0.4) when claiming new-kinase prediction.
3. Always pass **`--ape-resi-floor 9999`** for full non-loop features
   (script defaults are often 624 = N-lobe only).
4. Stage5/6 imputation means are **train-only** (aligned with AE μ/σ hygiene).
5. Stage10–12 may still assume older coordinate conventions; audit before
   claiming numbers.

---

## Counts to assert

| Checkpoint | Expect |
|------------|--------|
| `manifest_v9.csv` | 5,318 |
| `v9_addendum_merged.csv` | 1,213 |
| `manifest_v91.csv` / latent rows | **6,531** |
| Unique PDBs | 4,533 |
| Conserved non-loop residues (6UANC) | 128 → 8,128 pairs → **7,455** at ≥75% coverage |
| Chains after Stage5 coverage filter | **6,523** (drop ~8 high-impute) |

---
