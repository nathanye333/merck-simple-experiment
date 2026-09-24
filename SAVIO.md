# Using Savio for this project

Savio is Berkeley’s Linux HPC cluster. In this repo it is mainly for **dockstring docking against F2**. Native Windows cannot run dockstring’s AutoDock Vina binaries.

Official docs: [Savio user guide](https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/), [Data Discovery computing](https://datadisco.cdss.berkeley.edu/).

## Where to run what

| Step | Tool | Where | Why |
|------|------|--------|-----|
| EDA / train ChemProp | notebooks, GPU | **Local Windows** (or Savio GPU later) | Heavy Python stack already set up in `.venv-chemprop` |
| LLM propose SMILES | NRP LLM API | **Local** (`propose_and_dock_f2.py`) | Needs `.env` / API key; no cluster needed |
| ChemProp predict | `predict_chemprop_f2.py` | **Local** | Checkpoint lives under `models/`; GPU optional |
| Dock F2 (oracle) | `dock_f2.py` / dockstring | **Savio compute node** | Linux + Open Babel (`obabel`) + Vina |
| Compare scores | JSON / notes | Either | Lower kcal/mol is better |

**MVP pattern:** propose + ChemProp on Windows → copy SMILES → dock on Savio → compare to DOCKSTRING F2 context (median ≈ −8.2, top 10% ≈ −9.4).

Do **not** dock on Savio login nodes (`ln00x`). Use `srun` / `sbatch` compute nodes.

## Account / partition (this project)

From `sacctmgr`, Data Discovery access often looks like:

- Account: `ic_cdss170fall` (confirm with `sacctmgr -p show associations user=$USER`)
- Useful partitions: `savio3_htc`, `savio2_htc` (CPU; enough for docking)
- QoS examples: `savio_normal`, `savio_debug` (short jobs)

Docking one molecule needs **CPU only** (no GPU partition).

## Login

From your laptop (PowerShell or terminal):

```bash
ssh YOUR_USERNAME@hpc.brc.berkeley.edu
```

Authenticate with PIN + Google Authenticator OTP.

Or use [Open OnDemand](https://ood.brc.berkeley.edu) → Clusters → BRC Shell Access.

`srun` / `module` / `conda` only work **after** you are on Savio — not in local PowerShell.

## One-time setup on Savio

### 1. Clone the repo

```bash
cd ~
git clone https://github.com/YOUR_USER/merck-simple-experiment.git
cd merck-simple-experiment
```

### 2. Conda env for docking

`dockstring` on conda-forge needs **Python &lt; 3.11**. Open Babel must provide the `obabel` CLI.

```bash
module unload python          # conflicts with anaconda3 if both loaded
module load anaconda3
conda config --set solver libmamba   # optional but faster

conda create -n dock -c conda-forge python=3.10 dockstring rdkit openbabel -y
source activate dock

which obabel    # must print a path
obabel -V
python -c "from dockstring import load_target; print('dockstring ok')"
```

If home quota is tight, create the env on scratch:

```bash
conda create -p /global/scratch/users/$USER/envs/dock \
  -c conda-forge python=3.10 dockstring rdkit openbabel -y
source activate /global/scratch/users/$USER/envs/dock
```

**Avoid for docking:** plain `pip` / `.venv` alone — easy to get dockstring without `obabel`. Prefer conda as above.

### 3. (Optional) API key on Savio

Only needed if you run `propose_and_dock_f2.py` on the cluster. Prefer keeping LLM + ChemProp local.

```bash
# on Savio, in the repo root — never commit this file
nano .env
# NRP_LLM_API_KEY=sk-...
chmod 600 .env
pip install python-dotenv openai   # inside the env that will run the script
```

## Get a compute node (interactive)

Still on the **login** node:

```bash
srun --account=ic_cdss170fall --partition=savio3_htc \
  --time=00:30:00 --cpus-per-task=4 --pty bash
```

- “Job queued and waiting for resources” = normal; wait for a compute prompt (hostname not `ln…`).
- Check status in another SSH session: `squeue -u $USER` (`PD` pending, `R` running).
- Ctrl-C cancels a waiting `srun`.
- If the queue is long, try `--partition=savio2_htc` or a debug QoS if your account allows it.

When the shell starts on a compute node:

```bash
module unload python
module load anaconda3
source activate dock
cd ~/merck-simple-experiment
```

## Dock a molecule

### Smoke test (argatroban in `dock_f2.py`)

```bash
python dock_f2.py
```

Expect **~30 s–3 min** (little progress output until it finishes). Prints best score (kcal/mol, lower better) and affinities.

### Dock an LLM candidate

Edit `SMILES` in `dock_f2.py`, or one-liner:

```bash
python - <<'PY'
from dockstring import load_target
smiles = "PASTE_STANDARDIZED_SMILES_HERE"
score, aux = load_target("F2").dock(smiles)
print("dock_score", score)
print("affinities", aux.get("affinities"))
PY
```

Paste the `smiles_std` from a local run under `data/llm_propose_dock/*_result.json`.

### Batch job (unattended)

`dock_job.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=dock_f2
#SBATCH --account=ic_cdss170fall
#SBATCH --partition=savio3_htc
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --output=dock_f2_%j.out

module unload python
module load anaconda3
source activate dock
cd ~/merck-simple-experiment
python dock_f2.py
```

```bash
sbatch dock_job.sh
squeue -u $USER
# when done:
cat dock_f2_JOBID.out
```

## Recommended split for this MVP

```text
Windows                          Savio (compute node)
─────────────────────────────    ────────────────────────────
propose_and_dock_f2.py
  --skip-dock
  → F2_pred + PNG + JSON
                                 dock_f2.py (or SMILES one-liner)
                                 → dock_score
Compare F2_pred vs dock_score vs DOCKSTRING percentiles
```

Local shortcuts:

```powershell
.\.venv-chemprop\Scripts\Activate.ps1
python propose_and_dock_f2.py --skip-dock
python predict_chemprop_f2.py --smiles "YOUR_SMILES"
```

## Leave Savio cleanly

| Goal | Command |
|------|---------|
| Leave compute node → back to login | `exit` (once) |
| Leave Savio → local PowerShell | `exit` again (or Ctrl-D) |
| Cancel queued/running job | `scancel JOBID` or Ctrl-C on waiting `srun` |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `srun` not found in PowerShell | You are local — `ssh` to Savio first |
| `ModuleNotFoundError: dockstring` | `source activate dock` (or install env above) |
| `FileNotFoundError: ... 'obabel'` | Install Open Babel in conda: `conda install -c conda-forge openbabel` |
| `python=3.11` + dockstring conflict | Use `python=3.10` |
| `anaconda3` won’t load with `python` | `module unload python` then `module load anaconda3` |
| Windows-style `.venv\Scripts\activate` | On Savio use `source .venv/bin/activate` — but prefer conda for docking |
| Docking on `ln00x` | Move to `srun` / `sbatch` compute node |
| Slow `pip` / `uv` on login | Use conda; put caches/envs on scratch; avoid huge installs on login |
| Job stuck `PD` forever | Try another partition; check `squeue -u $USER`; ask if account/QoS limits apply |

## Score context (F2 DOCKSTRING)

From project EDA (kcal/mol, lower = better):

| Percentile | Approx. score |
|------------|----------------|
| top 1% | −10.4 |
| top 10% | −9.4 |
| median | −8.2 |
| p90 (weak) | −6.9 |

ChemProp `F2_pred` is a 2D proxy; dockstring is the docking check. They need not match exactly — treat both as imperfect vs wet-lab affinity.
