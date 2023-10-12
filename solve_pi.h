extern const double dx;

// basic sequencial
double solve_sequencial_basic();

#ifdef SSE2
// simd 256
double solve_sequencial_simd_128();
#endif

#ifdef AVX
// simd 256
double solve_sequencial_simd_256();
#endif

#ifdef AVX512F
// simd 512
double solve_sequencial_simd_512();
#endif

// multi-thread basic
double solve_parallel_basic();

// multi-thread bad because of coherence miss
double solve_parallel_coherence_miss();

#ifdef AVX
// multi-thread simd
double solve_parallel_simd_256();
#endif

#ifdef AVX512F
// multi-thread simd
double solve_parallel_simd_512();
#endif