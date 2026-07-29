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

#define DEFAULT_SIZE 2048U
#define DEFAULT_ITERATIONS 500U
#define DEFAULT_THREADS 4U

typedef struct {
    size_t n;
    size_t iterations;
    size_t threads;
    double *src;
    double *dst;
    pthread_barrier_t barrier;
} stencil_context_t;

typedef struct {
    size_t tid;
    stencil_context_t *context;
} worker_arg_t;

static void usage(const char *program)
{
    fprintf(stderr,
            "Usage: %s [-n grid_size] [-i iterations] [-t threads]\n"
            "Defaults: -n %u -i %u -t %u\n",
            program, DEFAULT_SIZE, DEFAULT_ITERATIONS, DEFAULT_THREADS);
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

static double *allocate_grid(size_t elements)
{
    void *memory = NULL;

    if (posix_memalign(&memory, 64, elements * sizeof(double)) != 0) {
        return NULL;
    }
    return (double *)memory;
}

static void *jacobi_rows(void *opaque)
{
    worker_arg_t *arg = (worker_arg_t *)opaque;
    stencil_context_t *context = arg->context;
    const size_t interior_rows = context->n - 2;
    const size_t row_begin = 1 + interior_rows * arg->tid / context->threads;
    const size_t row_end = 1 + interior_rows * (arg->tid + 1) /
                                  context->threads;
    size_t iteration;

    for (iteration = 0; iteration < context->iterations; ++iteration) {
        double *src = context->src;
        double *dst = context->dst;
        size_t i;

        for (i = row_begin; i < row_end; ++i) {
            const size_t offset = i * context->n;
            size_t j;

            for (j = 1; j + 1 < context->n; ++j) {
                const size_t index = offset + j;
                dst[index] = 0.2 *
                             (src[index] + src[index - 1] + src[index + 1] +
                              src[index - context->n] +
                              src[index + context->n]);
            }
        }

        pthread_barrier_wait(&context->barrier);
        if (arg->tid == 0) {
            double *temporary = context->src;
            context->src = context->dst;
            context->dst = temporary;
        }
        pthread_barrier_wait(&context->barrier);
    }

    return NULL;
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
    size_t iterations = DEFAULT_ITERATIONS;
    size_t threads = DEFAULT_THREADS;
    size_t elements;
    stencil_context_t context;
    pthread_t *workers = NULL;
    worker_arg_t *args = NULL;
    struct timespec start;
    struct timespec finish;
    long double checksum = 0.0L;
    double boundary_error = 0.0;
    int barrier_initialized = 0;
    int option;
    int status = EXIT_FAILURE;
    size_t i;

    memset(&context, 0, sizeof(context));
    while ((option = getopt(argc, argv, "n:i:t:h")) != -1) {
        switch (option) {
        case 'n':
            if (parse_positive(optarg, &n) != 0) {
                usage(argv[0]);
                return EXIT_FAILURE;
            }
            break;
        case 'i':
            if (parse_positive(optarg, &iterations) != 0) {
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

    if (n < 3 || threads > n - 2 || threads > 256) {
        fprintf(stderr,
                "STENCIL requires size >= 3 and 1 <= threads <= "
                "min(size - 2, 256).\n");
        return EXIT_FAILURE;
    }
    if (n > SIZE_MAX / n) {
        fprintf(stderr, "STENCIL grid size is too large.\n");
        return EXIT_FAILURE;
    }
    elements = n * n;
    if (elements > SIZE_MAX / sizeof(double)) {
        fprintf(stderr, "STENCIL allocation size overflows size_t.\n");
        return EXIT_FAILURE;
    }

    context.n = n;
    context.iterations = iterations;
    context.threads = threads;
    context.src = allocate_grid(elements);
    context.dst = allocate_grid(elements);
    workers = calloc(threads > 1 ? threads - 1 : 1, sizeof(*workers));
    args = calloc(threads, sizeof(*args));
    if (context.src == NULL || context.dst == NULL || workers == NULL ||
        args == NULL) {
        fprintf(stderr, "STENCIL could not allocate its working set.\n");
        goto cleanup;
    }

    for (i = 0; i < elements; ++i) {
        const size_t row = i / n;
        const size_t column = i % n;
        const double background =
            (double)((row * 13U + column * 7U) % 101U) / 101.0;
        const double hot_spot =
            (row > n / 3 && row < (2 * n) / 3 &&
             column > n / 3 && column < (2 * n) / 3)
                ? 1.0
                : 0.0;
        context.src[i] = background + hot_spot;
        context.dst[i] = context.src[i];
    }

    if (pthread_barrier_init(&context.barrier, NULL, (unsigned)threads) != 0) {
        fprintf(stderr, "STENCIL could not initialize its barrier.\n");
        goto cleanup;
    }
    barrier_initialized = 1;
    for (i = 0; i < threads; ++i) {
        args[i].tid = i;
        args[i].context = &context;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (i = 1; i < threads; ++i) {
        const int error = pthread_create(&workers[i - 1], NULL,
                                         jacobi_rows, &args[i]);
        if (error != 0) {
            fprintf(stderr, "STENCIL pthread_create failed: %s\n",
                    strerror(error));
            exit(EXIT_FAILURE);
        }
    }
    jacobi_rows(&args[0]);
    for (i = 1; i < threads; ++i) {
        pthread_join(workers[i - 1], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);

    for (i = 0; i < elements; ++i) {
        if (!isfinite(context.src[i])) {
            fprintf(stderr, "STENCIL validation failed: non-finite value.\n");
            goto cleanup;
        }
        checksum += (long double)context.src[i];
    }
    for (i = 0; i < n; ++i) {
        boundary_error = fmax(boundary_error,
                              fabs(context.src[i] - context.dst[i]));
        boundary_error = fmax(
            boundary_error,
            fabs(context.src[(n - 1) * n + i] -
                 context.dst[(n - 1) * n + i]));
        boundary_error = fmax(boundary_error,
                              fabs(context.src[i * n] - context.dst[i * n]));
        boundary_error = fmax(
            boundary_error,
            fabs(context.src[i * n + n - 1] -
                 context.dst[i * n + n - 1]));
    }
    if (boundary_error != 0.0L) {
        fprintf(stderr, "STENCIL validation failed: boundary changed.\n");
        goto cleanup;
    }

    printf("STENCIL reproduction workload\n");
    printf("Grid size            : %zu x %zu\n", n, n);
    printf("Threads              : %zu\n", threads);
    printf("Iterations           : %zu\n", iterations);
    printf("Kernel time (seconds): %.6f\n", seconds_since(&start, &finish));
    printf("Checksum             : %.12Le\n", checksum);
    printf("Boundary error       : %.12e\n", boundary_error);
    printf("Validation           : PASS\n");
    status = EXIT_SUCCESS;

cleanup:
    if (barrier_initialized) {
        pthread_barrier_destroy(&context.barrier);
    }
    free(args);
    free(workers);
    free(context.dst);
    free(context.src);
    return status;
}
