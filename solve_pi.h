extern const double dx;

// basic sequencial
double solve_sequencial_basic();

// simd 256
double solve_sequencial_simd_256();

// simd 512
double solve_sequencial_simd_512();

// multi-thread basic
double solve_parallel_basic();

// multi-thread bad because of coherence miss
double solve_parallel_coherence_miss();

// multi-thread simd
double solve_parallel_simd_512();
