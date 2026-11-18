#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("logs", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
rows = {}
for path in args.logs.rglob("*"):
    if not path.is_file(): continue
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.strip().split(",")
        if len(parts) != 7 or parts[0] not in {"pthreads", "openmp", "mpi_c", "mpi_python"}: continue
        try:
            row = [parts[0], int(parts[1]), int(parts[2]), int(parts[3]),
                   float(parts[4]), float(parts[5]), float(parts[6])]
        except ValueError:
            continue
        key = (row[0], row[1], row[2], row[3])
        if key not in rows or row[4] < rows[key][4]: rows[key] = row
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["implementation", "n", "bandwidth", "workers", "time_s", "residual", "error"])
    writer.writerows(rows[k] for k in sorted(rows))
print(f"wrote {len(rows)} rows to {args.output}")
