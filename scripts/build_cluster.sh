#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load compiler/gcc/11
module load mpi/openmpi/4.1.6/gcc/11
module load python/3.10_conda
make clean
make -j3

if ! python3 -c 'import mpi4py' >/dev/null 2>&1; then
  python3 -m venv --system-site-packages .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  MPICC=mpicc python -m pip install --no-binary=mpi4py mpi4py
fi

echo "Build complete."
gcc --version | head -1
mpirun --version | head -1
python3 --version
