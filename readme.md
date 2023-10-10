# Use SIMD and OpenMP to optimize your C code

1. Make sure your cpu is of x86-64 architecture and supports avx and avx512f extensions.(Use `lscpu` to check.)
2. You can modify the OPTIMIZATION in `makefile` to find out how gcc speed up your code without your manual optimization.(Default level is `-O0`, i.e. no optimization)
3. All solving functions use Numerical Integration Method to solve pi, so changing the `dx` in `main.cpp` will result in the precision of calculated pi. The smaller dx is, the more precise pi is.
4. Pay attention to Coherence Miss! Though sometimes your parallel code accesses different variables, but as long as they share the same one cache block, cache miss will also occurs.(Also use `lscpu` to check your cache block size, i.e. `cache_alignment`) So make sure never have a processor share any space on the same block.
