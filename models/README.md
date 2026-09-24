# Frozen ChemProp F2 models (checkpoints are gitignored; meta JSON is tracked)

After a full-data train in `train_chemprop_f2.ipynb` (§7), you should have:

- `chemprop_f2_full_cluster.ckpt` — Lightning / ChemProp weights
- `chemprop_f2_full_cluster_meta.json` — radius, top RDKit list, metrics

Predict without retraining:

```bash
.\.venv-chemprop\Scripts\Activate.ps1
python predict_chemprop_f2.py --smiles "CCO"
```
