#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("csv_file", type=Path)
parser.add_argument("output_dir", type=Path)
args = parser.parse_args()
series = defaultdict(list)
with args.csv_file.open() as f:
    for row in csv.DictReader(f):
        series[row["implementation"]].append((int(row["workers"]), float(row["time_s"])))
args.output_dir.mkdir(parents=True, exist_ok=True)
labels = {"pthreads":"Pthreads", "openmp":"OpenMP", "mpi_c":"C + MPI", "mpi_python":"Python + MPI"}

fig, ax = plt.subplots(figsize=(7.2, 4.2))
for name, values in sorted(series.items()):
    values.sort(); p = [v[0] for v in values]; t = [v[1] for v in values]
    ax.plot(p, t, marker="o", label=labels.get(name, name))
ax.set_xlabel("Число потоков / процессов"); ax.set_ylabel("Время, с")
ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
fig.savefig(args.output_dir / "time.pdf"); fig.savefig(args.output_dir / "time.png", dpi=180); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.2))
for name, values in sorted(series.items()):
    values.sort(); p = [v[0] for v in values]; t = [v[1] for v in values]
    base = next((tt for pp, tt in values if pp == 1), t[0])
    ax.plot(p, [base / tt for tt in t], marker="o", label=labels.get(name, name))
ax.set_xlabel("Число потоков / процессов"); ax.set_ylabel("Ускорение")
ax.grid(True, alpha=.3); ax.legend(); fig.tight_layout()
fig.savefig(args.output_dir / "speedup.pdf"); fig.savefig(args.output_dir / "speedup.png", dpi=180); plt.close(fig)
