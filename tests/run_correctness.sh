#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -f /etc/profile.d/modules.sh ]]; then
  source scripts/env_cluster.sh
fi

N=${N:-128}
BW=${BW:-5}
SEED=${SEED:-42}
TOL=${TOL:-1e-10}
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT

check() {
  local line="$1"
  echo "$line"
  python3 - "$line" "$TOL" <<'PY'
import sys
fields = sys.argv[1].strip().split(',')
assert len(fields) == 7, fields
residual, error, tol = float(fields[5]), float(fields[6]), float(sys.argv[2])
if residual > tol or error > tol:
    raise SystemExit(f"FAILED: residual={residual:g}, error={error:g}, tolerance={tol:g}")
PY
}

check "$(./bin/band_pthreads "$N" "$BW" "$SEED" 4 1)"
check "$(./bin/band_openmp "$N" "$BW" "$SEED" 4 1)"
if command -v mpirun >/dev/null; then
  check "$(mpirun --mca btl ^openib --oversubscribe -np 4 ./bin/band_mpi "$N" "$BW" "$SEED" 1)"
  if python3 -c 'import mpi4py, numpy' >/dev/null 2>&1; then
    check "$(mpirun --mca btl ^openib --oversubscribe -np 4 python3 src/band_mpi.py "$N" "$BW" "$SEED" 1)"
  else
    echo "SKIP mpi_python: mpi4py/numpy unavailable" >&2
  fi
fi
echo "All available implementations passed."
