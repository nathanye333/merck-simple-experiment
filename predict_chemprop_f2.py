"""Load a frozen ChemProp F2 model and predict docking scores (no retraining).

Example:
  python predict_chemprop_f2.py --smiles "CCO"
  python predict_chemprop_f2.py --smiles-file mols.txt --ckpt models/chemprop_f2_full_cluster.ckpt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

DEFAULT_CKPT = Path("models/chemprop_f2_full_cluster.ckpt")
DEFAULT_META = Path("models/chemprop_f2_full_cluster_meta.json")
DEFAULT_EDA_FEAT = Path("data/eda_feature_selection.json")

FALLBACK_TOP_RDKIT = [
    "BertzCT",
    "RingCount",
    "HeavyAtomCount",
    "NumAromaticRings",
    "NumAmideBonds",
    "SMR_VSA7",
    "NumRotatableBonds",
    "EState_VSA1",
    "NumAliphaticRings",
    "NumHDonors",
]

RDKIT_DESC_FUNCS = {
    "MolWt": Descriptors.MolWt,
    "HeavyAtomCount": Descriptors.HeavyAtomCount,
    "NumHAcceptors": Descriptors.NumHAcceptors,
    "NumHDonors": Descriptors.NumHDonors,
    "MolLogP": Descriptors.MolLogP,
    "TPSA": Descriptors.TPSA,
    "NumRotatableBonds": Descriptors.NumRotatableBonds,
    "NumAromaticRings": Descriptors.NumAromaticRings,
    "NumAliphaticRings": Descriptors.NumAliphaticRings,
    "RingCount": Descriptors.RingCount,
    "FractionCSP3": Descriptors.FractionCSP3,
    "NumHeteroatoms": Descriptors.NumHeteroatoms,
    "LabuteASA": Descriptors.LabuteASA,
    "BertzCT": Descriptors.BertzCT,
    "Chi0v": Descriptors.Chi0v,
    "Chi1v": Descriptors.Chi1v,
    "Kappa1": Descriptors.Kappa1,
    "Kappa2": Descriptors.Kappa2,
    "PEOE_VSA1": Descriptors.PEOE_VSA1,
    "PEOE_VSA6": Descriptors.PEOE_VSA6,
    "SMR_VSA1": Descriptors.SMR_VSA1,
    "SMR_VSA7": Descriptors.SMR_VSA7,
    "SlogP_VSA1": Descriptors.SlogP_VSA1,
    "SlogP_VSA8": Descriptors.SlogP_VSA8,
    "EState_VSA1": Descriptors.EState_VSA1,
    "VSA_EState1": Descriptors.VSA_EState1,
    "qed": QED.qed,
    "NumAmideBonds": rdMolDescriptors.CalcNumAmideBonds,
    "NumAtomStereoCenters": rdMolDescriptors.CalcNumAtomStereoCenters,
}

_uncharger = rdMolStandardize.Uncharger()
_lfc = rdMolStandardize.LargestFragmentChooser()


def standardize_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = _lfc.choose(mol)
    mol = _uncharger.uncharge(mol)
    return Chem.MolToSmiles(mol)


def rdkit_desc_vector(smi: str, keys: list[str]) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    vals = []
    for name in keys:
        fn = RDKIT_DESC_FUNCS[name]
        try:
            vals.append(float(fn(mol)))
        except Exception:
            return None
    if not np.all(np.isfinite(vals)):
        return None
    return np.asarray(vals, dtype=np.float64)


def load_meta(meta_path: Path | None = None, eda_path: Path = DEFAULT_EDA_FEAT) -> dict:
    meta_path = meta_path or DEFAULT_META
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    if eda_path.exists():
        eda = json.loads(eda_path.read_text(encoding="utf-8"))
        return {
            "morgan_radius": int(eda.get("best_morgan_radius_by_xgb", 2)),
            "top_rdkit_descriptors": eda.get("top_rdkit_descriptors", FALLBACK_TOP_RDKIT),
            "mol_feature_set": "top_rdkit",
            "use_mol_features": True,
        }
    return {
        "morgan_radius": 2,
        "top_rdkit_descriptors": FALLBACK_TOP_RDKIT,
        "mol_feature_set": "top_rdkit",
        "use_mol_features": True,
    }


def load_model(ckpt_path: Path | str = DEFAULT_CKPT):
    import torch
    from chemprop.models import MPNN

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {ckpt_path}\n"
            "Train the full-data model in train_chemprop_f2.ipynb (§7 export) first."
        )
    # Checkpoint may have been saved on CUDA (e.g. Windows GPU). Map to CPU when no CUDA
    # so the same predict path works on Mac / CPU-only machines. Trainer(accelerator="auto")
    # still places the model on the best available device at predict time.
    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PyTorch 2.6+ defaults weights_only=True; ChemProp ckpts pickle metric classes (e.g. RMSE).
    # ChemProp.load_from_checkpoint rewrites the ckpt then Lightning loads again — patch both paths.
    _orig_load = torch.load

    def _trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        # ChemProp often passes map_location positionally: torch.load(path, map_location, ...)
        # Only inject CPU mapping when the caller left it unset.
        if not torch.cuda.is_available():
            if len(args) >= 2:
                if args[1] is None:
                    args = (args[0], map_location, *args[2:])
            elif kwargs.get("map_location") is None:
                kwargs["map_location"] = map_location
        return _orig_load(*args, **kwargs)

    torch.load = _trusted_load  # type: ignore[assignment]
    try:
        return MPNN.load_from_checkpoint(
            str(ckpt_path), map_location=map_location, weights_only=False
        )
    finally:
        torch.load = _orig_load  # type: ignore[assignment]


def predict_smiles(
    smiles: list[str] | str,
    ckpt_path: Path | str = DEFAULT_CKPT,
    meta_path: Path | str | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Predict F2 docking scores for SMILES. Returns a dataframe with smiles_in, smiles_std, F2_pred."""
    from lightning import pytorch as pl
    from chemprop import data, featurizers

    if isinstance(smiles, str):
        smiles = [smiles]

    meta = load_meta(Path(meta_path) if meta_path else None)
    use_mol = bool(meta.get("use_mol_features", True))
    feat_set = meta.get("mol_feature_set", "top_rdkit")
    top_keys = list(meta.get("top_rdkit_descriptors", FALLBACK_TOP_RDKIT))

    rows = []
    datapoints = []
    kept_idx = []
    for i, smi in enumerate(smiles):
        std = standardize_smiles(smi)
        if std is None:
            rows.append({"smiles_in": smi, "smiles_std": None, "F2_pred": np.nan, "error": "invalid_smiles"})
            continue
        x_d = None
        if use_mol and feat_set == "top_rdkit":
            x_d = rdkit_desc_vector(std, top_keys)
            if x_d is None:
                rows.append(
                    {
                        "smiles_in": smi,
                        "smiles_std": std,
                        "F2_pred": np.nan,
                        "error": "rdkit_featurize_failed",
                    }
                )
                continue
        elif use_mol and feat_set != "top_rdkit":
            raise NotImplementedError(
                f"predict_chemprop_f2 currently supports mol_feature_set='top_rdkit' or use_mol_features=False; got {feat_set!r}"
            )

        if x_d is None:
            dp = data.MoleculeDatapoint.from_smi(std, np.array([0.0]))
        else:
            dp = data.MoleculeDatapoint.from_smi(std, np.array([0.0]), x_d=x_d)
        datapoints.append(dp)
        kept_idx.append(len(rows))
        rows.append({"smiles_in": smi, "smiles_std": std, "F2_pred": np.nan, "error": None})

    if not datapoints:
        return pd.DataFrame(rows)

    model = load_model(ckpt_path)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset = data.MoleculeDataset(datapoints, featurizer)
    # Do NOT normalize X_d here — checkpoint ScaleTransform scales raw extras at predict time.
    # ChemProp drops the last batch when len % batch_size == 1 (breaks n=1); keep batch_size safe.
    eff_batch = min(batch_size, len(datapoints))
    if len(datapoints) % eff_batch == 1:
        eff_batch = max(1, eff_batch - 1) if eff_batch > 1 else 1
    loader = data.build_dataloader(dset, batch_size=eff_batch, num_workers=0, shuffle=False)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
    )
    pred_batches = trainer.predict(model, dataloaders=loader)

    preds = []
    for batch in pred_batches:
        if hasattr(batch, "detach"):
            preds.append(batch.detach().cpu().numpy().reshape(-1))
        elif isinstance(batch, (list, tuple)):
            preds.append(np.concatenate([np.asarray(x).reshape(-1) for x in batch]))
        else:
            preds.append(np.asarray(batch).reshape(-1))
    y_pred = np.concatenate(preds)
    if len(y_pred) != len(kept_idx):
        raise RuntimeError(f"pred length {len(y_pred)} != kept {len(kept_idx)}")

    for j, row_i in enumerate(kept_idx):
        rows[row_i]["F2_pred"] = float(y_pred[j])

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict F2 docking scores with a frozen ChemProp model")
    parser.add_argument("--smiles", nargs="*", default=None, help="One or more SMILES strings")
    parser.add_argument("--smiles-file", type=Path, default=None, help="Text file with one SMILES per line")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT, help="Lightning / ChemProp checkpoint")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Model metadata JSON")
    parser.add_argument("--out", type=Path, default=None, help="Optional CSV output path")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    smiles: list[str] = []
    if args.smiles:
        smiles.extend(args.smiles)
    if args.smiles_file is not None:
        smiles.extend(
            [
                line.strip()
                for line in args.smiles_file.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        )
    if not smiles:
        parser.error("Provide --smiles and/or --smiles-file")

    df = predict_smiles(smiles, ckpt_path=args.ckpt, meta_path=args.meta, batch_size=args.batch_size)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"wrote {args.out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    # Avoid OpenMP abort on some Windows conda/pip mixes
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    torch.set_float32_matmul_precision("high")
    main()
