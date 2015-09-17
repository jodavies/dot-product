# dot-product

How fast can we compute the dot product of two (large) vectors? By *large*, we mean *many times larger than L3 cache*. The performance will therefore be limited by how quickly we can access DRAM.

Systems using dual-channel DDR(2,3,4)-N have a peak theoretical memory bandwidth of N * (64 bits) * (2 channels) / 8 / 1024 GiB/s.

In all cases, we use the highest possible optimization level of the compiler (`-Ofast` for `gcc`, `-O3` for `icc`) but *disallow* floating-point unsafe optimizations (`-fno-unsafe-math-optimizations` for `gcc`, `-fp-model precise` for `icc`). This will severely limit the optimizations the compiler can perform on this code, since it cannot re-order floating point calculations. When allowing such optimizations makes a large difference, it will be discussed. Additionally, tests are run with as few other programs running as possible, including the desktop environment. This improves performance by a few percent, and decreaces the variance of the measurements.




## Array Sum
Begin by considering the simpler case of summing the elements of a single large array. Results for `gcc` are as graphed (`icc` produces very similar performance), with some explanation of each test below.

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



#### How does DRAM work?
To make any further progress, we need to examine carefully how exactly DRAM works.

The DRAM in the test systems is *dual rank*. Each rank consists of 8 *banks*, which contain an array of bytes. When we read from DRAM, we read 1KB from each bank, forming an 8KB *page*. This page is transferred to *sense amplifiers* from which the data is transferred to the CPU. Further accesses from the same page are now cheap, as it does not have to be loaded into the sense amps again. *(This is why sequential access is faster than random access -- we don't have to load a new page so often). (This is ALSO why the CPU's hardware prefetcher only prefetches within the open 8KB page -- swapping pages to prefetch further could reduce performance).*

The two ranks can be accessed in parallel. We therefore want to read data from multiple pages at the same time. We can re-structure the loop such that we access the array in this way. The code now has fairly strict requirements on the size of the array we are summing (it must be a multiple of concurrent pages * memory page size) but this is just a proof-of-concept.

The reason that accessing *4* pages seems to improve performance is that we have 2 memory *channels* -- each with its own DIMM of two ranks. Above 4 pages, performance should drop.?????? DOES IT? YES UNLESS PREFETCHING.



#### Software Prefetching
To try and further saturate the memory controller we can insert software prefetch instructions into the loops. When the loop reaches the prefetched location, it will find the data in L1 cache and not have to request it from DRAM.

WE WERE ALREADY LOADING FROM DRAM EVERY CYCLE, BY HIDING THE ADDITION LATENCY AS MUCH AS POSSIBLE. WHY SHOULD PREFETCHING HELP?
