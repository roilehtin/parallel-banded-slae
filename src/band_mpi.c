#include <mpi.h>
#include "band_common.h"

static int solve_mpi(BandMatrix *a, double *b, double *x, int rank, int size) {
    double *pivot_row = (double *)malloc((size_t)a->width * sizeof(double));
    if (!pivot_row) return -1;
    for (int k = 0; k < a->n - 1; ++k) {
        int owner = k % size;
        if (rank == owner) memcpy(pivot_row, a->data + (size_t)k * a->width,
                                  (size_t)a->width * sizeof(double));
        MPI_Bcast(pivot_row, a->width, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        MPI_Bcast(&b[k], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        double pivot = pivot_row[a->bw];
        if (fabs(pivot) < 1.0e-14) { free(pivot_row); return -1; }
        int imax = k + a->bw < a->n - 1 ? k + a->bw : a->n - 1;
        int jmax = k + a->bw < a->n - 1 ? k + a->bw : a->n - 1;
        int first = k + 1;
        int remainder = first % size;
        if (remainder != rank) first += (rank - remainder + size) % size;
        for (int i = first; i <= imax; i += size) {
            double factor = band_get(a, i, k) / pivot;
            band_set(a, i, k, 0.0);
            for (int j = k + 1; j <= jmax; ++j) {
                int pd = j - k + a->bw;
                band_set(a, i, j, band_get(a, i, j) - factor * pivot_row[pd]);
            }
            b[i] -= factor * b[k];
        }
    }
    for (int k = a->n - 1; k >= 0; --k) {
        int owner = k % size;
        if (rank == owner) {
            double s = b[k];
            int jmax = k + a->bw < a->n - 1 ? k + a->bw : a->n - 1;
            for (int j = k + 1; j <= jmax; ++j) s -= band_get(a, k, j) * x[j];
            x[k] = s / band_get(a, k, k);
        }
        MPI_Bcast(&x[k], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
    }
    free(pivot_row); return 0;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (argc < 4 || argc > 5) {
        if (rank == 0) fprintf(stderr, "usage: %s N BW SEED [REPEATS]\n", argv[0]);
        MPI_Finalize(); return 2;
    }
    int n = parse_int(argv[1], 2, "N"), bw = parse_int(argv[2], 1, "BW");
    int seed = parse_int(argv[3], 0, "SEED");
    int repeats = argc == 5 ? parse_int(argv[4], 1, "REPEATS") : 1;
    if (n < 0 || bw < 0 || seed < 0 || repeats < 0 || bw >= n) { MPI_Finalize(); return 2; }
    BandMatrix base; double *base_b, *ref;
    if (generate_problem(&base, &base_b, &ref, n, bw, (uint64_t)seed) != 0) {
        MPI_Finalize(); return 2;
    }
    double best = HUGE_VAL, *answer = (double *)calloc((size_t)n, sizeof(double));
    for (int r = 0; r < repeats; ++r) {
        BandMatrix a; band_init(&a, n, bw);
        double *b = (double *)malloc((size_t)n * sizeof(double));
        memcpy(a.data, base.data, (size_t)n * (size_t)a.width * sizeof(double));
        memcpy(b, base_b, (size_t)n * sizeof(double));
        memset(answer, 0, (size_t)n * sizeof(double));
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        int rc = solve_mpi(&a, b, answer, rank, size);
        double local = MPI_Wtime() - t0, elapsed;
        MPI_Reduce(&local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank == 0 && elapsed < best) best = elapsed;
        band_free(&a); free(b);
        if (rc != 0) { MPI_Finalize(); return 3; }
    }
    if (rank == 0) print_result("mpi_c", n, bw, size, best,
        relative_residual(&base, base_b, answer), relative_error(answer, ref, n));
    free(answer); free(base_b); free(ref); band_free(&base);
    MPI_Finalize(); return 0;
}
