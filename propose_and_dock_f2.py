"""Propose a thrombin (F2) binder via NRP LLM, score with ChemProp, dock with dockstring.

Uses the OpenAI-compatible NRP endpoint (https://datadisco.cdss.berkeley.edu/nrp-llm-api/).
API key: NRP_LLM_API_KEY in .env (never commit).

ChemProp 2 needs Python >=3.11; conda-forge dockstring needs Python <3.11. Run this
script in the ChemProp env and pass --dock-python (or DOCK_PYTHON) to a dockstring
env interpreter so one CLI still returns F2_pred + dock_score.

Example:
  python propose_and_dock_f2.py
  python propose_and_dock_f2.py --model gpt-oss --skip-dock
  python propose_and_dock_f2.py --smiles "CCO"   # skip LLM; still predict + dock
  python propose_and_dock_f2.py --dock-python /path/to/f2-mvp/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from rdkit import Chem
from rdkit.Chem import Draw

from predict_chemprop_f2 import (
    DEFAULT_CKPT,
    DEFAULT_META,
    predict_smiles,
    standardize_smiles,
)

NRP_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
DEFAULT_MODEL = "qwen3"

# Rough DOCKSTRING F2 context from EDA (kcal/mol; lower = better)
F2_CONTEXT = {
    "p1": -10.4,
    "p10": -9.4,
    "median": -8.2,
    "p90": -6.9,
}

SYSTEM_PROMPT = """You are a medicinal chemist designing a drug-like small molecule that might bind \
coagulation factor II (thrombin / F2).

Rules:
- Reply with EXACTLY one valid RDKit-parseable SMILES string on a single line.
- No markdown, no explanation, no name, no quotes — only the SMILES.
- Prefer drug-like size (roughly 250–550 Da), avoid exotic elements and disconnected fragments.
- Do not copy a known commercial drug SMILES verbatim; propose a plausible analog or novel idea.
"""

USER_PROMPT = (
    "Propose one SMILES for a thrombin (F2) binder candidate. "
    "Output only the SMILES string."
)


def load_nrp_client() -> OpenAI:
    load_dotenv(Path(".env"))
    api_key = (os.getenv("NRP_LLM_API_KEY") or "").strip().strip('"').strip("'")
    if not api_key:
        raise SystemExit(
            "Missing NRP_LLM_API_KEY. Put it in .env (gitignored) or the environment.\n"
            "See https://datadisco.cdss.berkeley.edu/nrp-llm-api/"
        )
    return OpenAI(api_key=api_key, base_url=NRP_BASE_URL)


def extract_smiles(text: str) -> str | None:
    """Pull a candidate SMILES from model output (handles accidental wrappers)."""
    if not text:
        return None
    raw = text.strip()
    # Strip common fences / quotes
    raw = re.sub(r"^```(?:smiles|chemistry)?\s*", "", raw, flags=re.I)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip().strip("`").strip('"').strip("'")
    # Prefer first non-empty line that parses
    for line in raw.splitlines():
        cand = line.strip().strip("`").strip('"').strip("'")
        if not cand or cand.lower().startswith(("smiles", "molecule")):
            continue
        # Drop leading labels like "SMILES:"
        cand = re.sub(r"^(?:smiles\s*[:=]\s*)", "", cand, flags=re.I)
        if standardize_smiles(cand) is not None:
            return cand
    # Last resort: first whitespace-free token that looks like SMILES
    tokens = re.findall(r"[A-Za-z0-9@+\-\[\]\(\)=#$:/\\.\\\\%]+", raw)
    for tok in tokens:
        if len(tok) >= 3 and standardize_smiles(tok) is not None:
            return tok
    return None


def propose_smiles(
    client: OpenAI,
    model: str,
    max_attempts: int = 3,
) -> tuple[str, str, str]:
    """Return (raw_smiles, standardized_smiles, raw_llm_text). Retries on invalid SMILES."""
    last_raw = ""
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        user = USER_PROMPT
        if feedback:
            user = (
                f"{USER_PROMPT}\n\nPrevious reply was not a valid SMILES ({feedback}). "
                "Try again with only a valid SMILES."
            )
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.8,
        )
        last_raw = (completion.choices[0].message.content or "").strip()
        cand = extract_smiles(last_raw)
        if cand is None:
            feedback = f"could not parse: {last_raw[:120]!r}"
            print(f"[attempt {attempt}/{max_attempts}] invalid LLM output: {feedback}")
            continue
        std = standardize_smiles(cand)
        if std is None:
            feedback = f"RDKit rejected: {cand!r}"
            print(f"[attempt {attempt}/{max_attempts}] {feedback}")
            continue
        return cand, std, last_raw
    raise RuntimeError(f"LLM did not return a valid SMILES after {max_attempts} attempts. Last: {last_raw!r}")


_DOCK_SUBPROCESS_CODE = r"""
import json, sys
from dockstring import load_target
smiles = sys.argv[1]
score, aux = load_target("F2").dock(smiles)
aux = aux or {}
print(json.dumps({"score": score, "affinities": aux.get("affinities")}))
"""


def dock_f2(
    smiles: str,
    dock_python: str | Path | None = None,
) -> tuple[float | None, dict]:
    """Dock SMILES against F2.

    If dock_python is set, run dockstring in that interpreter (separate env).
    Otherwise import dockstring in-process.
    """
    import platform

    # dockstring ships AutoDock Vina binaries for Linux/macOS only
    if platform.system() == "Windows":
        raise RuntimeError(
            "dockstring docking is not supported on native Windows "
            "(no vina_windows binary). Run this step under WSL/Linux, DataHub, or SAVIO."
        )

    if dock_python:
        return _dock_f2_subprocess(smiles, Path(dock_python))

    try:
        from dockstring import load_target
    except ImportError as exc:
        raise RuntimeError(
            "dockstring is not importable in this env. "
            "Pass --dock-python /path/to/dockstring-env/bin/python "
            "(or set DOCK_PYTHON) so docking runs in a Python <3.11 env."
        ) from exc

    target = load_target("F2")
    score, aux = target.dock(smiles)
    return score, aux or {}


def _dock_f2_subprocess(smiles: str, dock_python: Path) -> tuple[float | None, dict]:
    if not dock_python.is_file():
        raise FileNotFoundError(f"dock python not found: {dock_python}")

    # Calling env/bin/python without `micromamba activate` leaves env/bin off PATH,
    # so dockstring cannot find `obabel`. Prepend the interpreter's bin dir.
    env = os.environ.copy()
    dock_bin = str(dock_python.resolve().parent)
    env["PATH"] = dock_bin + os.pathsep + env.get("PATH", "")

    proc = subprocess.run(
        [str(dock_python), "-c", _DOCK_SUBPROCESS_CODE, smiles],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"exit {proc.returncode}"
        raise RuntimeError(f"dock subprocess failed ({dock_python}): {err}")

    # Last non-empty line should be the JSON payload (dockstring may print logs).
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"dock subprocess returned no stdout ({dock_python})")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"dock subprocess did not return JSON (last line={lines[-1]!r})"
        ) from exc

    return payload.get("score"), {"affinities": payload.get("affinities")}


def score_context(score: float | None) -> str:
    if score is None:
        return "no score"
    # Lower is better
    if score <= F2_CONTEXT["p1"]:
        return f"~top 1% of DOCKSTRING F2 (p1~{F2_CONTEXT['p1']})"
    if score <= F2_CONTEXT["p10"]:
        return f"~top 10% of DOCKSTRING F2 (p10~{F2_CONTEXT['p10']})"
    if score <= F2_CONTEXT["median"]:
        return f"better than median DOCKSTRING F2 (median~{F2_CONTEXT['median']})"
    if score <= F2_CONTEXT["p90"]:
        return f"weaker than median DOCKSTRING F2 (median~{F2_CONTEXT['median']})"
    return f"weak vs DOCKSTRING F2 (p90~{F2_CONTEXT['p90']})"


def save_structure_png(smiles: str, path: Path) -> None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Draw.MolToFile(mol, str(path), size=(400, 300))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="LLM-propose a molecule, ChemProp-predict F2, dockstring-dock F2"
    )
    parser.add_argument("--smiles", default=None, help="Skip LLM; use this SMILES instead")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"NRP chat model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--meta", type=Path, default=DEFAULT_META)
    parser.add_argument("--max-attempts", type=int, default=3, help="LLM retries if SMILES invalid")
    parser.add_argument("--skip-dock", action="store_true", help="ChemProp only (faster)")
    parser.add_argument(
        "--dock-python",
        type=Path,
        default=None,
        help=(
            "Python interpreter with dockstring (e.g. micromamba env f2-mvp). "
            "Defaults to $DOCK_PYTHON. Use when ChemProp and dockstring cannot share an env."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/llm_propose_dock"),
        help="Directory for JSON / PNG outputs",
    )
    args = parser.parse_args(argv)

    dock_python = args.dock_python
    if dock_python is None:
        env_dock = (os.getenv("DOCK_PYTHON") or "").strip()
        if env_dock:
            dock_python = Path(env_dock)

    llm_raw = None
    model_used = None
    if args.smiles:
        smiles_in = args.smiles
        smiles_std = standardize_smiles(smiles_in)
        if smiles_std is None:
            raise SystemExit(f"Invalid SMILES: {smiles_in!r}")
        print(f"Using provided SMILES (std): {smiles_std}")
    else:
        client = load_nrp_client()
        model_used = args.model
        print(f"Requesting SMILES from NRP model={model_used!r} ...")
        smiles_in, smiles_std, llm_raw = propose_smiles(
            client, model=model_used, max_attempts=args.max_attempts
        )
        print(f"LLM SMILES (raw): {smiles_in}")
        print(f"Standardized:     {smiles_std}")

    print("\nChemProp F2 prediction...")
    pred_df = predict_smiles([smiles_std], ckpt_path=args.ckpt, meta_path=args.meta)
    f2_pred = pred_df.iloc[0]["F2_pred"]
    pred_err = pred_df.iloc[0].get("error")
    if pred_err:
        print(f"ChemProp warning: {pred_err}")
    print(f"F2_pred (kcal/mol, lower better): {f2_pred}")
    if f2_pred == f2_pred:  # not NaN
        print(f"  context vs DOCKSTRING: {score_context(float(f2_pred))}")

    dock_score = None
    affinities = None
    if args.skip_dock:
        print("\nSkipping dockstring (--skip-dock).")
    else:
        if dock_python:
            print(f"\nDocking via subprocess ({dock_python}) against F2...")
        else:
            print("\nDocking with dockstring against F2 (this can take a minute)...")
        try:
            dock_score, aux = dock_f2(smiles_std, dock_python=dock_python)
            affinities = aux.get("affinities")
            print(f"Docking score (kcal/mol, lower better): {dock_score}")
            print(f"  context vs DOCKSTRING: {score_context(dock_score)}")
            if affinities is not None:
                print(f"  affinities: {affinities}")
        except Exception as exc:
            print(f"Docking failed: {exc}")
            print(
                "  (ChemProp prediction above is still valid; "
                "pass --dock-python to a dockstring env or fix in-process dockstring.)"
            )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{run_id}_mol.png"
    json_path = out_dir / f"{run_id}_result.json"
    save_structure_png(smiles_std, png_path)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "smiles_in": smiles_in,
        "smiles_std": smiles_std,
        "llm_model": model_used,
        "llm_raw": llm_raw,
        "F2_pred": None if f2_pred != f2_pred else float(f2_pred),
        "chemprop_error": None if pred_err is None or (isinstance(pred_err, float) and pred_err != pred_err) else str(pred_err),
        "dock_score": dock_score,
        "affinities": affinities,
        "dock_python": str(dock_python) if dock_python else sys.executable,
        "F2_context_kcal_mol": F2_CONTEXT,
        "ckpt": str(args.ckpt),
        "structure_png": str(png_path) if png_path.exists() else None,
        "notes": (
            "Primary label is DOCKSTRING F2 docking score (kcal/mol, lower better). "
            "ChemProp is a 2D predictor; dockstring is the docking oracle for this MVP. "
            "Docking is imperfect — treat as a proxy, not wet-lab affinity."
        ),
    }
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n--- summary ---")
    print(f"SMILES:    {smiles_std}")
    print(f"F2_pred:   {result['F2_pred']}")
    print(f"dock:      {dock_score}")
    print(f"structure: {png_path if png_path.exists() else '(not written)'}")
    print(f"saved:     {json_path}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import torch

    torch.set_float32_matmul_precision("high")
    main()
