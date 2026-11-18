#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=112
#SBATCH --ntasks-per-node=56
#SBATCH --partition=tornado
#SBATCH --time=00:30:00
#SBATCH --job-name=band_mpi_c
#SBATCH --output=logs/mpi-c-%j.out
#SBATCH --error=logs/mpi-c-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source scripts/env_cluster.sh
mkdir -p logs
N=${N:-6000}; BW=${BW:-96}; SEED=${SEED:-42}; REPEATS=${REPEATS:-3}
echo "implementation,n,bw,workers,time_s,residual,error"
for p in 1 2 4 8 16 28 56 84 112; do
  mpirun --mca btl ^openib -np "$p" --map-by hwthread --bind-to hwthread \
    ./bin/band_mpi "$N" "$BW" "$SEED" "$REPEATS"
done
