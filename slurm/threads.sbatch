#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --partition=cascade
#SBATCH --time=00:30:00
#SBATCH --job-name=band_threads
#SBATCH --output=logs/threads-%j.out
#SBATCH --error=logs/threads-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/env_cluster.sh
mkdir -p logs
N=${N:-6000}; BW=${BW:-96}; SEED=${SEED:-42}; REPEATS=${REPEATS:-3}
echo "implementation,n,bw,workers,time_s,residual,error"
for p in 1 2 4 8 12 16 24 32 48; do
  ./bin/band_pthreads "$N" "$BW" "$SEED" "$p" "$REPEATS"
done
for p in 1 2 4 8 12 16 24 32 48; do
  OMP_PROC_BIND=close OMP_PLACES=cores ./bin/band_openmp "$N" "$BW" "$SEED" "$p" "$REPEATS"
done
