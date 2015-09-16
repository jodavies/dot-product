# dot-product

How fast can we compute the dot product of two (large) vectors? By *large*, we mean *many times larger than L3 cache*. The performance will therefore be limited by how quickly we can access DRAM.

Systems using dual-channel DDR(2,3,4)-N have a peak theoretical memory bandwidth of N * (64 bits) * (2 channels) / 8 / 1024 GiB/s.

In all cases, we use the highest possible optimization level of the compiler (`-Ofast` for `gcc`, `-O3` for `icc`) but *disallow* floating-point unsafe optimizations (`-fno-unsafe-math-optimizations` for `gcc`, `-fp-model precise` for `icc`). This will severely limit the optimizations the compiler can perform on this code, since it cannot re-order floating point calculations. When allowing such optimizations makes a large difference, it will be discussed.




## Array Sum
Begin by considering the simpler case of summing the elements of a single large array. Results are as pictured, with some explanation of each test below:

![desktop](img/desktop.png)



#### Simple
Here we simply loop over `sum += array[i]` (unrolled by two cache lines). This produces by far the worst performing code. It is not vectorized by `gcc` or `icc` (as expected) and acheives only ~45% peak memory bandwidth.

*(If we allow unsafe float optimizations, however, things improve a lot. We now see over 70% peak bandwidth. `gcc` and `icc` use packed AVX instructions to add 4 doubles in parallel. This somewhat equivalent to using 4 partial sums, see below.)*




#### Partial Sums
The problem with the previous method is that we have a single summation variable, to which we add with a loop carried dependence. It takes 8 sequential additions (24 cycles) to exhaust the current cache-line (of 64 bytes), and we then have to load another from DRAM. It would be preferrable to load a cache-line from DRAM *every* loop iteration. We can achieve this by using more summation variables to hide the addition latency. Four variables are sufficient on modern hardware. Using the loop
```c
for (int i = 0; i < arraySize; i+=8) {
	sum0 += array[i+0];
	sum1 += array[i+1];
	sum2 += array[i+2];
	sum3 += array[i+3];
	sum0 += array[i+4];
	sum1 += array[i+5];
	sum2 += array[i+6];
	sum3 += array[i+7];
}
```
we sum an entire cache line with only 8 cycles spent in addition.

If the compiler manages to vectorize the code, two (or four) of the summation variables can be combined into a single SSE (or AVX) vector, and the number of partial sums is no longer sufficient to hide the addition latency.

*(Here, neither `gcc` nor `icc` vectorize the loop without unsafe float optimizations, despite that fact that it can be done without re-ordering any float operations).*



#### Explicit Vectorization
We now turn to vector intrinsics to explicitly vectorize the sum. This means we'll have vectorized code without enabling unsafe float optimizations, and the produced code will be much cleaner than autovectorized code. We try both SSE and AVX intrinsics.

Vectorized code means that we spend fewer cycles on addition per array element, since modern (AVX supporting) CPUs can add 4 doubles in a single instruction. At this point, we have already considered the latency of the addition operations and tried to cover it using partial sums, so we expect only a *minimal improvement* by vectorizing.

At this point, enabling unsafe float optimizations make no difference, or make the code *slower*.



#### Software Prefetching
To try and further saturate the memory controller we can insert software prefetch instructions into the loops. When the loop reaches the prefetched location, it will find the data in L1 cache and not have to request it from DRAM.

WE WERE ALREADY LOADING FROM DRAM EVERY CYCLE. WHY SHOULD PREFETCHING HELP?





