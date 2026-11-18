#ifndef BAND_COMMON_H
#define BAND_COMMON_H

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    int n;
    int bw;
    int width;
    double *data;
} BandMatrix;

static inline double wall_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

static inline int band_init(BandMatrix *m, int n, int bw) {
    m->n = n;
    m->bw = bw;
    m->width = 2 * bw + 1;
    m->data = (double *)calloc((size_t)n * (size_t)m->width, sizeof(double));
    return m->data ? 0 : -1;
}

static inline void band_free(BandMatrix *m) {
    free(m->data);
    m->data = NULL;
}

static inline double band_get(const BandMatrix *m, int i, int j) {
    int d = j - i + m->bw;
    if (i < 0 || i >= m->n || j < 0 || j >= m->n || d < 0 || d >= m->width) {
        return 0.0;
    }
    return m->data[(size_t)i * (size_t)m->width + (size_t)d];
}

static inline void band_set(BandMatrix *m, int i, int j, double value) {
    int d = j - i + m->bw;
    if (i >= 0 && i < m->n && j >= 0 && j < m->n && d >= 0 && d < m->width) {
        m->data[(size_t)i * (size_t)m->width + (size_t)d] = value;
    }
}

static inline uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static inline double offdiag_value(int i, int j, int bw, uint64_t seed) {
    int lo = i < j ? i : j;
    int hi = i < j ? j : i;
    uint64_t key = seed ^ ((uint64_t)(uint32_t)lo << 32) ^ (uint32_t)hi;
    double u = (double)(mix64(key) % UINT64_C(1000000)) / 1000000.0;
    return -(0.05 + 0.20 * u) / (double)(bw > 0 ? bw : 1);
}

static inline double exact_x(int i) {
    return 1.0 + (double)(i % 17) / 17.0;
}

static inline int generate_problem(BandMatrix *a, double **b_out, double **x_out,
                                   int n, int bw, uint64_t seed) {
    if (n < 2 || bw < 1 || bw >= n || band_init(a, n, bw) != 0) return -1;
    double *b = (double *)calloc((size_t)n, sizeof(double));
    double *x = (double *)malloc((size_t)n * sizeof(double));
    if (!b || !x) {
        free(b); free(x); band_free(a); return -1;
    }
    for (int i = 0; i < n; ++i) {
        int jmax = i + bw < n - 1 ? i + bw : n - 1;
        for (int j = i + 1; j <= jmax; ++j) {
            double v = offdiag_value(i, j, bw, seed);
            band_set(a, i, j, v);
            band_set(a, j, i, v);
        }
    }
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        int j0 = i - bw > 0 ? i - bw : 0;
        int j1 = i + bw < n - 1 ? i + bw : n - 1;
        for (int j = j0; j <= j1; ++j) if (j != i) row_sum += fabs(band_get(a, i, j));
        band_set(a, i, i, row_sum + 1.0);
        x[i] = exact_x(i);
    }
    for (int i = 0; i < n; ++i) {
        int j0 = i - bw > 0 ? i - bw : 0;
        int j1 = i + bw < n - 1 ? i + bw : n - 1;
        for (int j = j0; j <= j1; ++j) b[i] += band_get(a, i, j) * x[j];
    }
    *b_out = b;
    *x_out = x;
    return 0;
}

static inline int back_substitution(const BandMatrix *a, const double *b, double *x) {
    for (int i = a->n - 1; i >= 0; --i) {
        double s = b[i];
        int j1 = i + a->bw < a->n - 1 ? i + a->bw : a->n - 1;
        for (int j = i + 1; j <= j1; ++j) s -= band_get(a, i, j) * x[j];
        double pivot = band_get(a, i, i);
        if (fabs(pivot) < 1.0e-14) return -1;
        x[i] = s / pivot;
    }
    return 0;
}

static inline double relative_error(const double *x, const double *ref, int n) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < n; ++i) {
        double d = x[i] - ref[i];
        num += d * d;
        den += ref[i] * ref[i];
    }
    return sqrt(num / den);
}

static inline double relative_residual(const BandMatrix *a, const double *b, const double *x) {
    double num = 0.0, den = 0.0;
    for (int i = 0; i < a->n; ++i) {
        double ax = 0.0;
        int j0 = i - a->bw > 0 ? i - a->bw : 0;
        int j1 = i + a->bw < a->n - 1 ? i + a->bw : a->n - 1;
        for (int j = j0; j <= j1; ++j) ax += band_get(a, i, j) * x[j];
        double d = ax - b[i];
        num += d * d;
        den += b[i] * b[i];
    }
    return sqrt(num / den);
}

static inline int parse_int(const char *text, int min_value, const char *name) {
    char *end = NULL;
    errno = 0;
    long v = strtol(text, &end, 10);
    if (errno || !end || *end || v < min_value || v > 100000000L) {
        fprintf(stderr, "invalid %s: %s\n", name, text);
        return -1;
    }
    return (int)v;
}

static inline void print_result(const char *impl, int n, int bw, int workers,
                                double seconds, double residual, double error) {
    printf("%s,%d,%d,%d,%.9f,%.6e,%.6e\n",
           impl, n, bw, workers, seconds, residual, error);
}

#endif
