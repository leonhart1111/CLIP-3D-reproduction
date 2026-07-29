#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define DEFAULT_SIZE 1024U
#define DEFAULT_REPEATS 1U
#define DEFAULT_THREADS 4U

typedef struct {
    size_t tid;
    size_t threads;
    size_t n;
    size_t repeats;
    const double *a;
    const double *b;
    double *c;
    double guard;
} worker_arg_t;

static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s [-n matrix_size] [-r repeats] [-t threads]\n"
            "Defaults: -n %u -r %u -t %u\n",
            program, DEFAULT_SIZE, DEFAULT_REPEATS, DEFAULT_THREADS);
}

static int parse_positive(const char *text, size_t *value)
{
    char *end = NULL;
    unsigned long long parsed;

    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0' || parsed == 0 ||
        parsed > SIZE_MAX) {
        return -1;
    }
    *value = (size_t)parsed;
    return 0;
}

static double *allocate_matrix(size_t elements)
{
    void *memory = NULL;

    if (posix_memalign(&memory, 64, elements * sizeof(double)) != 0) {
        return NULL;
    }
    return (double *)memory;
}

static void *multiply_rows(void *opaque)
{
    worker_arg_t *arg = (worker_arg_t *)opaque;
    const size_t row_begin = arg->n * arg->tid / arg->threads;
    const size_t row_end = arg->n * (arg->tid + 1) / arg->threads;
    size_t repeat;

    for (repeat = 0; repeat < arg->repeats; ++repeat) {
        size_t i;

        for (i = row_begin; i < row_end; ++i) {
            double *c_row = &arg->c[i * arg->n];
            size_t k;

            memset(c_row, 0, arg->n * sizeof(double));
            for (k = 0; k < arg->n; ++k) {
                const double a_value = arg->a[i * arg->n + k];
                const double *b_row = &arg->b[k * arg->n];
                size_t j;

                for (j = 0; j < arg->n; ++j) {
                    c_row[j] += a_value * b_row[j];
                }
            }
        }

        /* This observable read ensures every requested repeat is retained. */
        if (row_begin < row_end) {
            arg->guard += arg->c[row_begin * arg->n + repeat % arg->n];
        }
    }

    return NULL;
}

static int verify_samples(const double *a, const double *b, const double *c,
                          size_t n)
{
    const size_t rows[] = {0, n / 4, n / 2, (3 * n) / 4, n - 1};
    const size_t cols[] = {0, n / 3, n / 2, (2 * n) / 3, n - 1};
    size_t sample;

    for (sample = 0; sample < sizeof(rows) / sizeof(rows[0]); ++sample) {
        const size_t i = rows[sample];
        const size_t j = cols[sample];
        long double reference = 0.0L;
        long double error;
        long double scale;
        size_t k;

        for (k = 0; k < n; ++k) {
            reference += (long double)a[i * n + k] *
                         (long double)b[k * n + j];
        }
        error = fabsl((long double)c[i * n + j] - reference);
        scale = fmaxl(1.0L, fabsl(reference));
        if (!isfinite(c[i * n + j]) || error / scale > 1.0e-11L) {
            fprintf(stderr,
                    "MATMUL validation failed at (%zu,%zu): got %.17g, "
                    "reference %.17Lg\n",
                    i, j, c[i * n + j], reference);
            return -1;
        }
    }
    return 0;
}

static double seconds_since(const struct timespec *start,
                            const struct timespec *finish)
{
    return (double)(finish->tv_sec - start->tv_sec) +
           (double)(finish->tv_nsec - start->tv_nsec) * 1.0e-9;
}

int main(int argc, char **argv)
{
    size_t n = DEFAULT_SIZE;
    size_t repeats = DEFAULT_REPEATS;
    size_t threads = DEFAULT_THREADS;
    size_t elements;
    double *a = NULL;
    double *b = NULL;
    double *c = NULL;
    pthread_t *workers = NULL;
    worker_arg_t *args = NULL;
    struct timespec start;
    struct timespec finish;
    long double checksum = 0.0L;
    double guard = 0.0;
    int option;
    int status = EXIT_FAILURE;
    size_t i;

    while ((option = getopt(argc, argv, "n:r:t:h")) != -1) {
        switch (option) {
        case 'n':
            if (parse_positive(optarg, &n) != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            break;
        case 'r':
            if (parse_positive(optarg, &repeats) != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            break;
        case 't':
            if (parse_positive(optarg, &threads) != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            usage(argv[0]);
            return EXIT_SUCCESS;
        default:
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (threads > n || threads > 256) {
        fprintf(stderr, "MATMUL requires 1 <= threads <= min(size, 256).\n");
        return EXIT_FAILURE;
    }
    if (n > SIZE_MAX / n) {
        fprintf(stderr, "MATMUL matrix size is too large.\n");
        return EXIT_FAILURE;
    }
    elements = n * n;
    if (elements > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "MATMUL allocation size overflows size_t.\n");
        return EXIT_FAILURE;
    }

    a = allocate_matrix(elements);
    b = allocate_matrix(elements);
    c = allocate_matrix(elements);
    workers = calloc(threads > 1 ? threads - 1 : 1, sizeof(*workers));
    args = calloc(threads, sizeof(*args));
    if (a == NULL || b == NULL || c == NULL || workers == NULL ||
        args == NULL) {
        fprintf(stderr, "MATMUL could not allocate its working set.\n");
        goto cleanup;
    }

    for (i = 0; i < elements; ++i) {
        a[i] = (double)((i * 17U + 3U) % 251U) / 251.0;
        b[i] = (double)((i * 29U + 7U) % 257U) / 257.0;
        c[i] = 0.0;
    }

    for (i = 0; i < threads; ++i) {
        args[i].tid = i;
        args[i].threads = threads;
        args[i].n = n;
        args[i].repeats = repeats;
        args[i].a = a;
        args[i].b = b;
        args[i].c = c;
        args[i].guard = 0.0;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (i = 1; i < threads; ++i) {
        const int error = pthread_create(&workers[i - 1], NULL,
                                         multiply_rows, &args[i]);
        if (error != 0) {
            fprintf(stderr, "MATMUL pthread_create failed: %s\n",
                    strerror(error));
            exit(EXIT_FAILURE);
        }
    }
    multiply_rows(&args[0]);
    for (i = 1; i < threads; ++i) {
        pthread_join(workers[i - 1], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);

    if (verify_samples(a, b, c, n) != 0) {
        goto cleanup;
    }
    for (i = 0; i < elements; ++i) {
        checksum += (long double)c[i];
    }
    for (i = 0; i < threads; ++i) {
        guard += args[i].guard;
    }

    printf("MATMUL reproduction workload\n");
    printf("Matrix size          : %zu x %zu\n", n, n);
    printf("Threads              : %zu\n", threads);
    printf("Repeats              : %zu\n", repeats);
    printf("Kernel time (seconds): %.6f\n", seconds_since(&start, &finish));
    printf("Checksum             : %.12Le\n", checksum);
    printf("Repeat guard         : %.12e\n", guard);
    printf("Validation           : PASS\n");
    status = EXIT_SUCCESS;

cleanup:
    free(args);
    free(workers);
    free(c);
    free(b);
    free(a);
    return status;
}
