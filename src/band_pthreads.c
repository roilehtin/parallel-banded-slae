#include <pthread.h>
#include "band_common.h"

#ifdef __APPLE__
typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int trip;
    int generation;
} Barrier;

static int barrier_init(Barrier *b, int count) {
    b->count = 0; b->trip = count; b->generation = 0;
    return pthread_mutex_init(&b->mutex, NULL) || pthread_cond_init(&b->cond, NULL);
}

static void barrier_wait(Barrier *b) {
    pthread_mutex_lock(&b->mutex);
    int generation = b->generation;
    if (++b->count == b->trip) {
        b->count = 0;
        ++b->generation;
        pthread_cond_broadcast(&b->cond);
    } else {
        while (generation == b->generation) pthread_cond_wait(&b->cond, &b->mutex);
    }
    pthread_mutex_unlock(&b->mutex);
}

static void barrier_destroy(Barrier *b) {
    pthread_cond_destroy(&b->cond);
    pthread_mutex_destroy(&b->mutex);
}
#else
typedef pthread_barrier_t Barrier;
static int barrier_init(Barrier *b, int count) {
    return pthread_barrier_init(b, NULL, (unsigned)count);
}
static void barrier_wait(Barrier *b) {
    int rc = pthread_barrier_wait(b);
    if (rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD) abort();
}
static void barrier_destroy(Barrier *b) {
    pthread_barrier_destroy(b);
}
#endif

typedef struct {
    BandMatrix *a;
    double *b;
    Barrier barrier;
    int threads;
    int failed;
} Shared;

typedef struct { Shared *shared; int tid; } WorkerArg;

static void *eliminate_worker(void *ptr) {
    WorkerArg *arg = (WorkerArg *)ptr;
    Shared *s = arg->shared;
    BandMatrix *a = s->a;
    for (int k = 0; k < a->n - 1; ++k) {
        if (arg->tid == 0 && fabs(band_get(a, k, k)) < 1.0e-14) s->failed = 1;
        barrier_wait(&s->barrier);
        if (!s->failed) {
            int imax = k + a->bw < a->n - 1 ? k + a->bw : a->n - 1;
            int jmax = k + a->bw < a->n - 1 ? k + a->bw : a->n - 1;
            double pivot = band_get(a, k, k);
            for (int i = k + 1 + arg->tid; i <= imax; i += s->threads) {
                double factor = band_get(a, i, k) / pivot;
                band_set(a, i, k, 0.0);
                for (int j = k + 1; j <= jmax; ++j) {
                    band_set(a, i, j, band_get(a, i, j) - factor * band_get(a, k, j));
                }
                s->b[i] -= factor * s->b[k];
            }
        }
        barrier_wait(&s->barrier);
    }
    return NULL;
}

static int solve_pthreads(BandMatrix *a, double *b, double *x, int threads) {
    Shared shared = {.a = a, .b = b, .threads = threads, .failed = 0};
    if (barrier_init(&shared.barrier, threads) != 0) return -1;
    pthread_t *ids = (pthread_t *)malloc((size_t)threads * sizeof(pthread_t));
    WorkerArg *args = (WorkerArg *)malloc((size_t)threads * sizeof(WorkerArg));
    if (!ids || !args) { free(ids); free(args); barrier_destroy(&shared.barrier); return -1; }
    int created = 0;
    for (int t = 0; t < threads; ++t) {
        args[t].shared = &shared; args[t].tid = t;
        if (pthread_create(&ids[t], NULL, eliminate_worker, &args[t]) != 0) break;
        ++created;
    }
    if (created != threads) {
        fprintf(stderr, "pthread_create failed after %d threads\n", created);
        exit(2);
    }
    for (int t = 0; t < threads; ++t) pthread_join(ids[t], NULL);
    int rc = shared.failed ? -1 : back_substitution(a, b, x);
    free(ids); free(args); barrier_destroy(&shared.barrier);
    return rc;
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
        if (solve_pthreads(&a, b, answer, threads) != 0) return 3;
        double elapsed = wall_time() - t0;
        if (elapsed < best) best = elapsed;
        band_free(&a); free(b);
    }
    print_result("pthreads", n, bw, threads, best,
                 relative_residual(&base, base_b, answer), relative_error(answer, ref, n));
    free(answer); free(base_b); free(ref); band_free(&base); return 0;
}
