#include <immintrin.h>
#include "sum-routines.h"


// Naive routine. Unroll by two cache lines, so we can compare to later routines.
double Sum1(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	double sum = 0.0;

	#pragma omp parallel for default(none) shared(array) reduction(+:sum) schedule(static)
	for (int i = 0; i < arraySize; i+=16) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
#endif
		sum += array[i+0];
		sum += array[i+1];
		sum += array[i+2];
		sum += array[i+3];
		sum += array[i+4];
		sum += array[i+5];
		sum += array[i+6];
		sum += array[i+7];
		sum += array[i+8];
		sum += array[i+9];
		sum += array[i+10];
		sum += array[i+11];
		sum += array[i+12];
		sum += array[i+13];
		sum += array[i+14];
		sum += array[i+15];
	}

	return sum;
}



// Use partial sums. Test different numbers of them. Will need more than
// four if compiler manages to vectorize the loop with either SSE or AVX
// Unroll loop by same amount in all cases, 2 cache lines.
double Sum2p2(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	double sum0 = 0.0;
	double sum1 = 0.0;

	#pragma omp parallel for default(none) shared(array) reduction(+:sum0,sum1) schedule(static)
	for (int i = 0; i < arraySize; i+=16) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
#endif
		sum0 += array[i+0];
		sum1 += array[i+1];
		sum0 += array[i+2];
		sum1 += array[i+3];
		sum0 += array[i+4];
		sum1 += array[i+5];
		sum0 += array[i+6];
		sum1 += array[i+7];
		sum0 += array[i+8];
		sum1 += array[i+9];
		sum0 += array[i+10];
		sum1 += array[i+11];
		sum0 += array[i+12];
		sum1 += array[i+13];
		sum0 += array[i+14];
		sum1 += array[i+15];
	}

	return sum0 + sum1;
}
double Sum2p4(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	double sum0 = 0.0;
	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum3 = 0.0;

	#pragma omp parallel for default(none) shared(array) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int i = 0; i < arraySize; i+=16) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
#endif
		sum0 += array[i+0];
		sum1 += array[i+1];
		sum2 += array[i+2];
		sum3 += array[i+3];
		sum0 += array[i+4];
		sum1 += array[i+5];
		sum2 += array[i+6];
		sum3 += array[i+7];
		sum0 += array[i+8];
		sum1 += array[i+9];
		sum2 += array[i+10];
		sum3 += array[i+11];
		sum0 += array[i+12];
		sum1 += array[i+13];
		sum2 += array[i+14];
		sum3 += array[i+15];
	}

	return sum0 + sum1 + sum2 + sum3;
}
double Sum2p8(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	double sum0 = 0.0;
	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum3 = 0.0;
	double sum4 = 0.0;
	double sum5 = 0.0;
	double sum6 = 0.0;
	double sum7 = 0.0;

	#pragma omp parallel for default(none) shared(array) reduction(+:sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7) schedule(static)
	for (int i = 0; i < arraySize; i+=16) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
#endif
		sum0 += array[i+0];
		sum1 += array[i+1];
		sum2 += array[i+2];
		sum3 += array[i+3];
		sum4 += array[i+4];
		sum5 += array[i+5];
		sum6 += array[i+6];
		sum7 += array[i+7];
		sum0 += array[i+8];
		sum1 += array[i+9];
		sum2 += array[i+10];
		sum3 += array[i+11];
		sum4 += array[i+12];
		sum5 += array[i+13];
		sum6 += array[i+14];
		sum7 += array[i+15];
	}

	return sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7;
}
double Sum2p16(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	double sum0 = 0.0;
	double sum1 = 0.0;
	double sum2 = 0.0;
	double sum3 = 0.0;
	double sum4 = 0.0;
	double sum5 = 0.0;
	double sum6 = 0.0;
	double sum7 = 0.0;
	double sum8 = 0.0;
	double sum9 = 0.0;
	double sum10 = 0.0;
	double sum11 = 0.0;
	double sum12 = 0.0;
	double sum13 = 0.0;
	double sum14 = 0.0;
	double sum15 = 0.0;

	#pragma omp parallel for default(none) shared(array) reduction(+:sum0,sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8,sum9,sum10,sum11,sum12,sum13,sum14,sum15) schedule(static)
	for (int i = 0; i < arraySize; i+=16) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
#endif
		sum0 += array[i+0];
		sum1 += array[i+1];
		sum2 += array[i+2];
		sum3 += array[i+3];
		sum4 += array[i+4];
		sum5 += array[i+5];
		sum6 += array[i+6];
		sum7 += array[i+7];
		sum8 += array[i+8];
		sum9 += array[i+9];
		sum10 += array[i+10];
		sum11 += array[i+11];
		sum12 += array[i+12];
		sum13 += array[i+13];
		sum14 += array[i+14];
		sum15 += array[i+15];
	}

	return   sum0 + sum1 + sum2 + sum3 + sum4 + sum5 + sum6 + sum7
	       + sum8 + sum9 + sum10 + sum11 + sum12 + sum13 + sum14 + sum15;
}



// Explicit vectorization, to produce tighter code. Use SSE here.
double Sum3u1(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	__m128d sum0 = _mm_set1_pd(0.0);
	__m128d sum1 = _mm_set1_pd(0.0);
	__m128d sum2 = _mm_set1_pd(0.0);
	__m128d sum3 = _mm_set1_pd(0.0);
	__m128d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int i = 0; i < arraySize; i+=8) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance]), _MM_HINT_T0);
#endif
		a0 = _mm_load_pd(&array[i+0]);
		sum0 = _mm_add_pd(sum0, a0);
		a1 = _mm_load_pd(&array[i+2]);
		sum1 = _mm_add_pd(sum1, a1);
		a2 = _mm_load_pd(&array[i+4]);
		sum2 = _mm_add_pd(sum2, a2);
		a3 = _mm_load_pd(&array[i+6]);
		sum3 = _mm_add_pd(sum3, a3);
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1]
    ;
}
double Sum3u2(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	__m128d sum0 = _mm_set1_pd(0.0);
	__m128d sum1 = _mm_set1_pd(0.0);
	__m128d sum2 = _mm_set1_pd(0.0);
	__m128d sum3 = _mm_set1_pd(0.0);
	__m128d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int i = 0; i < arraySize; i+=16) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
#endif
		a0 = _mm_load_pd(&array[i+0]);
		sum0 = _mm_add_pd(sum0, a0);
		a1 = _mm_load_pd(&array[i+2]);
		sum1 = _mm_add_pd(sum1, a1);
		a2 = _mm_load_pd(&array[i+4]);
		sum2 = _mm_add_pd(sum2, a2);
		a3 = _mm_load_pd(&array[i+6]);
		sum3 = _mm_add_pd(sum3, a3);
		a0 = _mm_load_pd(&array[i+8]);
		sum0 = _mm_add_pd(sum0, a0);
		a1 = _mm_load_pd(&array[i+10]);
		sum1 = _mm_add_pd(sum1, a1);
		a2 = _mm_load_pd(&array[i+12]);
		sum2 = _mm_add_pd(sum2, a2);
		a3 = _mm_load_pd(&array[i+14]);
		sum3 = _mm_add_pd(sum3, a3);
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1]
    ;
}
double Sum3u4(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	__m128d sum0 = _mm_set1_pd(0.0);
	__m128d sum1 = _mm_set1_pd(0.0);
	__m128d sum2 = _mm_set1_pd(0.0);
	__m128d sum3 = _mm_set1_pd(0.0);
	__m128d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int i = 0; i < arraySize; i+=32) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+16]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+24]), _MM_HINT_T0);
#endif
		a0 = _mm_load_pd(&array[i+0]);
		sum0 = _mm_add_pd(sum0, a0);
		a1 = _mm_load_pd(&array[i+2]);
		sum1 = _mm_add_pd(sum1, a1);
		a2 = _mm_load_pd(&array[i+4]);
		sum2 = _mm_add_pd(sum2, a2);
		a3 = _mm_load_pd(&array[i+6]);
		sum3 = _mm_add_pd(sum3, a3);
		a0 = _mm_load_pd(&array[i+8]);
		sum0 = _mm_add_pd(sum0, a0);
		a1 = _mm_load_pd(&array[i+10]);
		sum1 = _mm_add_pd(sum1, a1);
		a2 = _mm_load_pd(&array[i+12]);
		sum2 = _mm_add_pd(sum2, a2);
		a3 = _mm_load_pd(&array[i+14]);
		sum3 = _mm_add_pd(sum3, a3);
		a0 = _mm_load_pd(&array[i+16]);
		sum0 = _mm_add_pd(sum0, a0);
		a1 = _mm_load_pd(&array[i+18]);
		sum1 = _mm_add_pd(sum1, a1);
		a2 = _mm_load_pd(&array[i+20]);
		sum2 = _mm_add_pd(sum2, a2);
		a3 = _mm_load_pd(&array[i+22]);
		sum3 = _mm_add_pd(sum3, a3);
		a0 = _mm_load_pd(&array[i+24]);
		sum0 = _mm_add_pd(sum0, a0);
		a1 = _mm_load_pd(&array[i+26]);
		sum1 = _mm_add_pd(sum1, a1);
		a2 = _mm_load_pd(&array[i+28]);
		sum2 = _mm_add_pd(sum2, a2);
		a3 = _mm_load_pd(&array[i+30]);
		sum3 = _mm_add_pd(sum3, a3);
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1]
    ;
}



// Explicit vectorization, to produce tighter code. Use AVX here.
double Sum4u1(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	__m256d sum0 = _mm256_set1_pd(0.0);
	__m256d sum1 = _mm256_set1_pd(0.0);
	__m256d sum2 = _mm256_set1_pd(0.0);
	__m256d sum3 = _mm256_set1_pd(0.0);
	__m256d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int i = 0; i < arraySize; i+=16) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
#endif
		a0 = _mm256_load_pd(&array[i+0]);
		sum0 = _mm256_add_pd(sum0, a0);
		a1 = _mm256_load_pd(&array[i+4]);
		sum1 = _mm256_add_pd(sum1, a1);
		a2 = _mm256_load_pd(&array[i+8]);
		sum2 = _mm256_add_pd(sum2, a2);
		a3 = _mm256_load_pd(&array[i+12]);
		sum3 = _mm256_add_pd(sum3, a3);
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1] + ((double*)&sum0)[2] + ((double*)&sum0)[3]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1] + ((double*)&sum1)[2] + ((double*)&sum1)[3]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1] + ((double*)&sum2)[2] + ((double*)&sum2)[3]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1] + ((double*)&sum3)[2] + ((double*)&sum3)[3]
    ;
}
double Sum4u2(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	__m256d sum0 = _mm256_set1_pd(0.0);
	__m256d sum1 = _mm256_set1_pd(0.0);
	__m256d sum2 = _mm256_set1_pd(0.0);
	__m256d sum3 = _mm256_set1_pd(0.0);
	__m256d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int i = 0; i < arraySize; i+=32) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+16]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+24]), _MM_HINT_T0);
#endif
		a0 = _mm256_load_pd(&array[i+0]);
		sum0 = _mm256_add_pd(sum0, a0);
		a1 = _mm256_load_pd(&array[i+4]);
		sum1 = _mm256_add_pd(sum1, a1);
		a2 = _mm256_load_pd(&array[i+8]);
		sum2 = _mm256_add_pd(sum2, a2);
		a3 = _mm256_load_pd(&array[i+12]);
		sum3 = _mm256_add_pd(sum3, a3);
		a0 = _mm256_load_pd(&array[i+16]);
		sum0 = _mm256_add_pd(sum0, a0);
		a1 = _mm256_load_pd(&array[i+20]);
		sum1 = _mm256_add_pd(sum1, a1);
		a2 = _mm256_load_pd(&array[i+24]);
		sum2 = _mm256_add_pd(sum2, a2);
		a3 = _mm256_load_pd(&array[i+28]);
		sum3 = _mm256_add_pd(sum3, a3);
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1] + ((double*)&sum0)[2] + ((double*)&sum0)[3]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1] + ((double*)&sum1)[2] + ((double*)&sum1)[3]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1] + ((double*)&sum2)[2] + ((double*)&sum2)[3]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1] + ((double*)&sum3)[2] + ((double*)&sum3)[3]
    ;
}
double Sum4u4(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{
	__m256d sum0 = _mm256_set1_pd(0.0);
	__m256d sum1 = _mm256_set1_pd(0.0);
	__m256d sum2 = _mm256_set1_pd(0.0);
	__m256d sum3 = _mm256_set1_pd(0.0);
	__m256d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int i = 0; i < arraySize; i+=64) {
#ifdef SWPREFETCH
		_mm_prefetch(&(array[i+prefetchDistance+0]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+8]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+16]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+24]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+32]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+40]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+48]), _MM_HINT_T0);
		_mm_prefetch(&(array[i+prefetchDistance+56]), _MM_HINT_T0);
#endif
		a0 = _mm256_load_pd(&array[i+0]);
		sum0 = _mm256_add_pd(sum0, a0);
		a1 = _mm256_load_pd(&array[i+4]);
		sum1 = _mm256_add_pd(sum1, a1);
		a2 = _mm256_load_pd(&array[i+8]);
		sum2 = _mm256_add_pd(sum2, a2);
		a3 = _mm256_load_pd(&array[i+12]);
		sum3 = _mm256_add_pd(sum3, a3);
		a0 = _mm256_load_pd(&array[i+16]);
		sum0 = _mm256_add_pd(sum0, a0);
		a1 = _mm256_load_pd(&array[i+20]);
		sum1 = _mm256_add_pd(sum1, a1);
		a2 = _mm256_load_pd(&array[i+24]);
		sum2 = _mm256_add_pd(sum2, a2);
		a3 = _mm256_load_pd(&array[i+28]);
		sum3 = _mm256_add_pd(sum3, a3);
		a0 = _mm256_load_pd(&array[i+32]);
		sum0 = _mm256_add_pd(sum0, a0);
		a1 = _mm256_load_pd(&array[i+36]);
		sum1 = _mm256_add_pd(sum1, a1);
		a2 = _mm256_load_pd(&array[i+40]);
		sum2 = _mm256_add_pd(sum2, a2);
		a3 = _mm256_load_pd(&array[i+44]);
		sum3 = _mm256_add_pd(sum3, a3);
		a0 = _mm256_load_pd(&array[i+48]);
		sum0 = _mm256_add_pd(sum0, a0);
		a1 = _mm256_load_pd(&array[i+52]);
		sum1 = _mm256_add_pd(sum1, a1);
		a2 = _mm256_load_pd(&array[i+56]);
		sum2 = _mm256_add_pd(sum2, a2);
		a3 = _mm256_load_pd(&array[i+60]);
		sum3 = _mm256_add_pd(sum3, a3);
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1] + ((double*)&sum0)[2] + ((double*)&sum0)[3]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1] + ((double*)&sum1)[2] + ((double*)&sum1)[3]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1] + ((double*)&sum2)[2] + ((double*)&sum2)[3]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1] + ((double*)&sum3)[2] + ((double*)&sum3)[3]
    ;
}



double Sum52(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{


	__m256d sum0 = _mm256_set1_pd(0.0);
	__m256d sum1 = _mm256_set1_pd(0.0);
	__m256d sum2 = _mm256_set1_pd(0.0);
	__m256d sum3 = _mm256_set1_pd(0.0);
	__m256d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int j = 0; j < arraySize; j+=2*MEMPAGESIZE) {
		for (int i = j; i < j+MEMPAGESIZE; i+= 16) {
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+0*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+0*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+1*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+1*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
		}
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1] + ((double*)&sum0)[2] + ((double*)&sum0)[3]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1] + ((double*)&sum1)[2] + ((double*)&sum1)[3]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1] + ((double*)&sum2)[2] + ((double*)&sum2)[3]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1] + ((double*)&sum3)[2] + ((double*)&sum3)[3]
    ;
}
double Sum54(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{


	__m256d sum0 = _mm256_set1_pd(0.0);
	__m256d sum1 = _mm256_set1_pd(0.0);
	__m256d sum2 = _mm256_set1_pd(0.0);
	__m256d sum3 = _mm256_set1_pd(0.0);
	__m256d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int j = 0; j < arraySize; j+=4*MEMPAGESIZE) {
		for (int i = j; i < j+MEMPAGESIZE; i+= 16) {
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+0*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+0*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+1*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+1*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+2*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+2*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+3*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+3*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
		}
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1] + ((double*)&sum0)[2] + ((double*)&sum0)[3]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1] + ((double*)&sum1)[2] + ((double*)&sum1)[3]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1] + ((double*)&sum2)[2] + ((double*)&sum2)[3]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1] + ((double*)&sum3)[2] + ((double*)&sum3)[3]
    ;
}
double Sum58(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{


	__m256d sum0 = _mm256_set1_pd(0.0);
	__m256d sum1 = _mm256_set1_pd(0.0);
	__m256d sum2 = _mm256_set1_pd(0.0);
	__m256d sum3 = _mm256_set1_pd(0.0);
	__m256d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int j = 0; j < arraySize; j+=8*MEMPAGESIZE) {
		for (int i = j; i < j+MEMPAGESIZE; i+= 16) {
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+0*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+0*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+1*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+1*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+2*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+2*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+3*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+3*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+4*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+4*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+4*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+4*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+4*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+4*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+5*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+5*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+5*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+5*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+5*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+5*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+6*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+6*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+6*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+6*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+6*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+6*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+7*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+7*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+7*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+7*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+7*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+7*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
		}
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1] + ((double*)&sum0)[2] + ((double*)&sum0)[3]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1] + ((double*)&sum1)[2] + ((double*)&sum1)[3]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1] + ((double*)&sum2)[2] + ((double*)&sum2)[3]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1] + ((double*)&sum3)[2] + ((double*)&sum3)[3]
    ;
}
double Sum516(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance)
{


	__m256d sum0 = _mm256_set1_pd(0.0);
	__m256d sum1 = _mm256_set1_pd(0.0);
	__m256d sum2 = _mm256_set1_pd(0.0);
	__m256d sum3 = _mm256_set1_pd(0.0);
	__m256d a0, a1, a2, a3;

	#pragma omp parallel for default(none) shared(array) private(a0,a1,a2,a3) reduction(+:sum0,sum1,sum2,sum3) schedule(static)
	for (int j = 0; j < arraySize; j+=16*MEMPAGESIZE) {
		for (int i = j; i < j+MEMPAGESIZE; i+= 16) {
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+0*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+0*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+1*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+1*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+2*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+2*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+2*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+3*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+3*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+3*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+4*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+4*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+4*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+4*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+4*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+4*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+5*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+5*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+5*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+5*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+5*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+5*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+6*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+6*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+6*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+6*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+6*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+6*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+7*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+7*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+7*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+7*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+7*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+7*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+8*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+8*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+8*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+8*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+8*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+8*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+9*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+9*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+9*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+9*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+9*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+9*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+10*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+10*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+10*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+10*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+10*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+10*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+11*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+11*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+11*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+11*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+11*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+11*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+12*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+12*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+12*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+12*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+12*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+12*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+13*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+13*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+13*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+13*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+13*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+13*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+14*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+14*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+14*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+14*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+14*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+14*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
#ifdef SWPREFETCH
			_mm_prefetch(&(array[i+15*MEMPAGESIZE+prefetchDistance+0]), _MM_HINT_T0);
			_mm_prefetch(&(array[i+15*MEMPAGESIZE+prefetchDistance+8]), _MM_HINT_T0);
#endif
			a0 = _mm256_load_pd(&array[i+15*MEMPAGESIZE+0]);
			sum0 = _mm256_add_pd(sum0, a0);
			a1 = _mm256_load_pd(&array[i+15*MEMPAGESIZE+4]);
			sum1 = _mm256_add_pd(sum1, a1);
			a2 = _mm256_load_pd(&array[i+15*MEMPAGESIZE+8]);
			sum2 = _mm256_add_pd(sum2, a2);
			a3 = _mm256_load_pd(&array[i+15*MEMPAGESIZE+12]);
			sum3 = _mm256_add_pd(sum3, a3);
		}
	}

	return   ((double*)&sum0)[0] + ((double*)&sum0)[1] + ((double*)&sum0)[2] + ((double*)&sum0)[3]
           + ((double*)&sum1)[0] + ((double*)&sum1)[1] + ((double*)&sum1)[2] + ((double*)&sum1)[3]
           + ((double*)&sum2)[0] + ((double*)&sum2)[1] + ((double*)&sum2)[2] + ((double*)&sum2)[3]
           + ((double*)&sum3)[0] + ((double*)&sum3)[1] + ((double*)&sum3)[2] + ((double*)&sum3)[3]
    ;
}

