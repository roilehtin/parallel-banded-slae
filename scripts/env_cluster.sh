#!/bin/bash
source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load compiler/gcc/11
module load mpi/openmpi/4.1.6/gcc/11
module load python/3.10_conda
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi
