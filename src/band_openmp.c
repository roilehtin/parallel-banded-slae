#include <omp.h>
#include "band_common.h"

static int solve_openmp(BandMatrix *a, double *b, double *x, int threads) {
    int failed = 0;
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
    #pragma omp parallel shared(failed, a, b)
    {
        for (int k = 0; k < a->n - 1; ++k) {
            #pragma omp single
            { if (fabs(band_get(a, k, k)) < 1.0e-14) failed = 1; }
            #pragma omp barrier
            int imax = k + a->bw < a->n - 1 ? k + a->bw : a->n - 1;
            int jmax = k + a->bw < a->n - 1 ? k + a->bw : a->n - 1;
            double pivot = band_get(a, k, k);
            #pragma omp for schedule(static)
            for (int i = k + 1; i <= imax; ++i) {
                if (!failed) {
                    double factor = band_get(a, i, k) / pivot;
                    band_set(a, i, k, 0.0);
                    for (int j = k + 1; j <= jmax; ++j)
                        band_set(a, i, j, band_get(a, i, j) - factor * band_get(a, k, j));
                    b[i] -= factor * b[k];
                }
            }
        }
    }
    return failed ? -1 : back_substitution(a, b, x);
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 6) {
        fprintf(stderr, "usage: %s N BW SEED THREADS [REPEATS]\n", argv[0]); return 2;
    }
    int n = parse_int(argv[1], 2, "N"), bw = parse_int(argv[2], 1, "BW");
    int seed = parse_int(argv[3], 0, "SEED"), threads = parse_int(argv[4], 1, "THREADS");
    int repeats = argc == 6 ? parse_int(argv[5], 1, "REPEATS") : 1;
    if (n < 0 || bw < 0 || seed < 0 || threads < 0 || repeats < 0 || bw >= n) return 2;
    BandMatrix base; double *base_b, *ref;
    if (generate_problem(&base, &base_b, &ref, n, bw, (uint64_t)seed) != 0) return 2;
    double best = HUGE_VAL, *answer = (double *)calloc((size_t)n, sizeof(double));
    for (int r = 0; r < repeats; ++r) {
        BandMatrix a; band_init(&a, n, bw);
        double *b = (double *)malloc((size_t)n * sizeof(double));
        memcpy(a.data, base.data, (size_t)n * (size_t)a.width * sizeof(double));
        memcpy(b, base_b, (size_t)n * sizeof(double));
        memset(answer, 0, (size_t)n * sizeof(double));
        double t0 = wall_time();
        if (solve_openmp(&a, b, answer, threads) != 0) return 3;
        double elapsed = wall_time() - t0;
        if (elapsed < best) best = elapsed;
        band_free(&a); free(b);
    }
    print_result("openmp", n, bw, threads, best,
                 relative_residual(&base, base_b, answer), relative_error(answer, ref, n));
    free(answer); free(base_b); free(ref); band_free(&base); return 0;
}
