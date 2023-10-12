#include <immintrin.h>
#include <xmmintrin.h>
#include <omp.h>
#include "solve_pi.h"

double solve_sequencial_basic() {
    double pi = 0;
    for (double x = 0.0; x < 1.0; x += dx) {
        pi += 1.0 / (1 + x * x);
    }
    pi *= 4 * dx;
    return pi;
}

#ifdef SSE2
double solve_sequencial_simd_128() {
    const int pack = sizeof(__m128d) / sizeof(double);

    __m128d SUM = _mm_set_pd(0, 0);
    __m128d X = _mm_set_pd(dx * 0, dx * 1);
    __m128d DX = _mm_set_pd(dx * pack, dx * pack);
    __m128d ONE = _mm_set_pd(1, 1);

    uint steps = (uint)(1.0 / dx) / pack;
    for (; steps--;) {
        SUM = _mm_add_pd(SUM, _mm_div_pd(ONE, _mm_add_pd(_mm_mul_pd(X, X), ONE)));
        X = _mm_add_pd(X, DX);
    }

    double sum[pack];
    // BUGFIX: _mm_store_pd require data to be aligned by 32 bytes!
    _mm_storeu_pd(sum, SUM);
    double pi = (sum[0] + sum[1]) * dx * 4;
    return pi;
}
#endif

#ifdef AVX
double solve_sequencial_simd_256() {
    const int pack = sizeof(__m256d) / sizeof(double);

    __m256d SUM = _mm256_set_pd(0, 0, 0, 0);
    __m256d X = _mm256_set_pd(dx * 0, dx * 1, dx * 2, dx * 3);
    __m256d DX = _mm256_set_pd(dx * pack, dx * pack, dx * pack, dx * pack);
    __m256d ONE = _mm256_set_pd(1, 1, 1, 1);

    uint steps = (uint)(1.0 / dx) / 4;
    for (; steps--;) {
        SUM = _mm256_add_pd(SUM, _mm256_div_pd(ONE, _mm256_add_pd(_mm256_mul_pd(X, X), ONE)));
        X = _mm256_add_pd(X, DX);
    }

    double sum[4];
    // BUGFIX: _mm256_store_pd require data to be aligned by 32 bytes!
    _mm256_storeu_pd(sum, SUM);
    double pi = (sum[0] + sum[1] + sum[2] + sum[3]) * dx * 4;
    return pi;
}
#endif

#ifdef AVX512F
double solve_sequencial_simd_512() {
    const int pack = sizeof(__m512d) / sizeof(double);

    __m512d SUM = _mm512_set_pd(0, 0, 0, 0, 0, 0, 0, 0);
    __m512d X = _mm512_set_pd(dx * 0, dx * 1, dx * 2, dx * 3, dx * 4, dx * 5, dx * 6, dx * 7);
    __m512d DX = _mm512_set_pd(dx * pack, dx * pack, dx * pack, dx * pack, dx * pack, dx * pack, dx * pack, dx * pack);
    __m512d ONE = _mm512_set_pd(1, 1, 1, 1, 1, 1, 1, 1);

    uint steps = (uint)(1.0 / dx) / 8;
    for (; steps--;) {
        SUM = _mm512_add_pd(SUM, _mm512_div_pd(ONE, _mm512_add_pd(_mm512_mul_pd(X, X), ONE)));
        X = _mm512_add_pd(X, DX);
    }

    double sum[8];
    // BUGFIX: _mm512_store_pd require data to be aligned by 32 bytes!
    _mm512_storeu_pd((void*)sum, SUM);
    double pi = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
    return pi * dx * 4;
}
#endif

double solve_parallel_basic() {
    double pi = 0;

    const int threads = 8;
    const double blocksize = 1.0 / threads;

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        double sum = 0;
        for (double x = blocksize * id; x < blocksize * (id + 1); x += dx) {
            sum += 1.0 / (1.0 + x * x);
        }
#pragma omp critical
        pi += sum * dx * 4;
    }

    return pi;
}

double solve_parallel_coherence_miss() {
    double pi = 0;
    double sum[8]{};

    const int threads = 8;
    const double blocksize = 1.0 / threads;

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (double x = blocksize * id; x < blocksize * (id + 1); x += dx) {
            // Because sum[0]...sum[7] are located in the same cache block, the cache block will 
            // ping-pongs between caches even though processors are accessing different variable.
            sum[id] += 1.0 / (1.0 + x * x);
        }
    }

    for (int i = 0; i < threads; ++i) {
        pi += sum[i] * dx * 4;
    }

    return pi;
}

#ifdef AVX512F
double solve_parallel_simd_512() {
    double pi = 0;

    const int pack = sizeof(__m512d) / sizeof(double);
    const int threads = 8;
    const double blocksize = 1.0 / threads;

    omp_set_num_threads(threads);
#pragma omp parallel
    {
        int id = omp_get_thread_num();

        double x0 = blocksize * id;
        __m512d SUM = _mm512_set_pd(0, 0, 0, 0, 0, 0, 0, 0);
        __m512d X = _mm512_set_pd(x0 + dx * 0, x0 + dx * 1, x0 + dx * 2, x0 + dx * 3, x0 + dx * 4, x0 + dx * 5, x0 + dx * 6, x0 + dx * 7);
        __m512d DX = _mm512_set_pd(dx * pack, dx * pack, dx * pack, dx * pack, dx * pack, dx * pack, dx * pack, dx * pack);
        __m512d ONE = _mm512_set_pd(1, 1, 1, 1, 1, 1, 1, 1);

        uint steps = (uint)(1.0 / dx / threads) / 8;
        for (; steps--;) {
            SUM = _mm512_add_pd(SUM, _mm512_div_pd(ONE, _mm512_add_pd(_mm512_mul_pd(X, X), ONE)));
            X = _mm512_add_pd(X, DX);
        }

        double sum[8];
        // BUGFIX: _mm512_store_pd require data to be aligned by 32 bytes!
        _mm512_storeu_pd((void*)sum, SUM);

        double s = sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7];
        s *= dx * 4;

#pragma omp critical
        pi += s;
    }

    return pi;
}
#endif