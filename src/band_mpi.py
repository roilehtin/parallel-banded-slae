#!/usr/bin/env python3
import math
import sys
import time
import numpy as np
from mpi4py import MPI

MASK = (1 << 64) - 1

def mix64(x: int) -> int:
    x &= MASK
    x ^= x >> 30; x = (x * 0xBF58476D1CE4E5B9) & MASK
    x ^= x >> 27; x = (x * 0x94D049BB133111EB) & MASK
    return (x ^ (x >> 31)) & MASK

def offdiag(i: int, j: int, bw: int, seed: int) -> float:
    lo, hi = min(i, j), max(i, j)
    key = (seed ^ (lo << 32) ^ hi) & MASK
    u = (mix64(key) % 1_000_000) / 1_000_000.0
    return -(0.05 + 0.20 * u) / max(bw, 1)

def generate(n: int, bw: int, seed: int):
    width = 2 * bw + 1
    a = np.zeros((n, width), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, min(n, i + bw + 1)):
            v = offdiag(i, j, bw, seed)
            a[i, j - i + bw] = v
            a[j, i - j + bw] = v
    a[:, bw] = np.sum(np.abs(a), axis=1) + 1.0
    x = 1.0 + (np.arange(n) % 17) / 17.0
    b = np.empty(n, dtype=np.float64)
    for i in range(n):
        j0, j1 = max(0, i - bw), min(n, i + bw + 1)
        b[i] = a[i, (j0 - i + bw):(j1 - i + bw)] @ x[j0:j1]
    return a, b, x

def solve(a, b, bw, comm):
    rank, size = comm.rank, comm.size
    n, width = a.shape
    x = np.zeros(n, dtype=np.float64)
    for k in range(n - 1):
        owner = k % size
        pivot_row = comm.bcast(a[k].copy() if rank == owner else None, root=owner)
        pivot_b = comm.bcast(float(b[k]) if rank == owner else None, root=owner)
        pivot = pivot_row[bw]
        if abs(pivot) < 1e-14:
            raise ArithmeticError(f"zero pivot at {k}")
        rows = np.arange(k + 1 + ((rank - (k + 1)) % size), min(n, k + bw + 1), size)
        if rows.size:
            factors = a[rows, k - rows + bw] / pivot
            js = np.arange(k + 1, min(n, k + bw + 1))
            cols = js[None, :] - rows[:, None] + bw
            a[rows[:, None], cols] -= factors[:, None] * pivot_row[js - k + bw][None, :]
            a[rows, k - rows + bw] = 0.0
            b[rows] -= factors * pivot_b
    for k in range(n - 1, -1, -1):
        owner = k % size
        value = None
        if rank == owner:
            j1 = min(n, k + bw + 1)
            value = (b[k] - a[k, bw + 1:bw + (j1 - k)] @ x[k + 1:j1]) / a[k, bw]
        x[k] = comm.bcast(value, root=owner)
    return x

def metrics(a, b, x, ref, bw):
    ax = np.empty_like(b)
    for i in range(len(b)):
        j0, j1 = max(0, i - bw), min(len(b), i + bw + 1)
        ax[i] = a[i, (j0 - i + bw):(j1 - i + bw)] @ x[j0:j1]
    return np.linalg.norm(ax - b) / np.linalg.norm(b), np.linalg.norm(x - ref) / np.linalg.norm(ref)

def main():
    comm = MPI.COMM_WORLD
    if len(sys.argv) not in (4, 5):
        if comm.rank == 0: print(f"usage: {sys.argv[0]} N BW SEED [REPEATS]", file=sys.stderr)
        return 2
    n, bw, seed = map(int, sys.argv[1:4]); repeats = int(sys.argv[4]) if len(sys.argv) == 5 else 1
    if n < 2 or not 1 <= bw < n or repeats < 1: return 2
    base_a, base_b, ref = generate(n, bw, seed)
    best, answer = math.inf, None
    for _ in range(repeats):
        a, b = base_a.copy(), base_b.copy()
        comm.Barrier(); start = time.perf_counter()
        answer = solve(a, b, bw, comm)
        elapsed = comm.reduce(time.perf_counter() - start, op=MPI.MAX, root=0)
        if comm.rank == 0: best = min(best, elapsed)
    if comm.rank == 0:
        residual, error = metrics(base_a, base_b, answer, ref, bw)
        print(f"mpi_python,{n},{bw},{comm.size},{best:.9f},{residual:.6e},{error:.6e}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
