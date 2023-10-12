#include <stdio.h>
#include <omp.h>
#include "solve_pi.h"

// the smaller dx is, the more precise pi is.
const double dx = 1e-9;

// wrapper function to calculate the time
void measure(const char *description, double (*p)(void)) {
    double t0, t1;
    t0 = omp_get_wtime();
    double pi = (*p)();
    t1 = omp_get_wtime();
    printf("%s:\n\tpi = %.10lf\n\ttime = %.3lfs\n", description, pi, t1 - t0);
}

int main() {
    measure("sequencial basic", &solve_sequencial_basic);
    #ifdef SSE2
    measure("sequencial simd 128", &solve_sequencial_simd_128);
    #endif
    #ifdef AVX
    measure("sequencial simd 256", &solve_sequencial_simd_256);
    #endif
    #ifdef AVX512F
    measure("sequencial simd 512", &solve_sequencial_simd_512);
    #endif
    measure("parallel basic (8 threads)", &solve_parallel_basic);
    #ifdef AVX
    measure("parallel simd 256 (8 threads)", &solve_parallel_simd_256);
    #endif
    #ifdef AVX512F
    measure("parallel simd 512 (8 threads)", &solve_parallel_simd_512);
    #endif
    measure("parallel slow with coherence miss (8 threads)", &solve_parallel_coherence_miss);
}
