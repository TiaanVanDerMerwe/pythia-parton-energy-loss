#!/bin/bash
#SBATCH --job-name=pythia_dihadron
#SBATCH --account=physics
#SBATCH --partition=ada
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/pythia_%A_%a.out
#SBATCH --error=logs/pythia_%A_%a.err
#SBATCH --mail-user=vmrtia003@myuct.ac.za
#SBATCH --mail-type=BEGIN,END,FAIL

module load software/pythia-8317b
export PYTHIA=/opt/exp_soft/pythia-8317pb
export LD_LIBRARY_PATH=$PYTHIA/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$PYTHIA/lib:$PYTHONPATH

# ── Parameters ───────────────────────────────────────────────────
COM=2760
NEVENTS=1000   # per job → 10M per bin after merging
N_SEEDS=100       # jobs per bin
POW=3

# ── Decode 2D index ──────────────────────────────────────────────
#   task 0–9   → bin 0, seeds 100–109
#   task 10–19 → bin 1, seeds 100–109  etc.
BIN=0
SEED=$(( SLURM_ARRAY_TASK_ID + 100 ))

mkdir -p logs
mkdir -p pythiaData/${COM}/cms

echo "[$BIN:$SEED] Started: $(date)"
echo "[$BIN:$SEED] Host:    $(hostname)"

python3 src/pythiaGenerator.py ${COM} ${NEVENTS} ${SEED} ${BIN}

echo "[$BIN:$SEED] Ended: $(date)"
