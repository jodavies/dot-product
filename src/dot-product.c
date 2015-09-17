// How quickly can we compute a dot product (single thread) of two vectors,
// which greatly exceed the CPU cache sizes? I.e., we'll be limited by
// the DRAM bandwidth of the system. Investigation into how we can maximise
// DRAM bandwidth use.
// Also test summing a single array.



// Test parameters:

// Comment out to disable testing of hugetlb/2MB pages. Need to already have these allocated by OS.
#define TESTHUGETLB 1

// Size of arrays used for testing. L3 caches are typically 6-8MiB, so
// choose, say, 25x this. The tests use double-prec floating point,
// (8B each).
#define ARRAYSIZE (25 * 8*1024*1024 /8)

// Number of times to run the bandwidth test.
#define NTIMES 100

// Theoretical peak memory bandwidth, to compute % achieved.
// For DDRX-N in dual channel mode, the peak bandwith is
// given by: N*64bits*2channels/8/1024 GiB/s
#define PEAKBW (1333.0*64.0*2.0/8.0/1024.0)

// Correct dot-product, using the array[i] = i initialization for both
// arrays. ( = sum_{i=0}^{ARRAYSIZE-1} i^2 )
#define CORRECTRESULT ((ARRAYSIZE-1.0)*((ARRAYSIZE-1.0)+1.0)*(2.0*(ARRAYSIZE-1.0)+1.0)/6.0)
// Correct sum, using the same initialization
#define CORRECTSUM ((ARRAYSIZE-1.0)*((ARRAYSIZE-1.0)+1.0)/2.0)

// Restrict and const keywords
#define RESTRICT restrict
#define CONST const



#include <stdio.h>
#include <stdlib.h>
#include <time.h>		//gettimeofday()
#include <math.h>		// sqrt()
#include <float.h>		// DBL_MIN
#include <immintrin.h>
#include <sys/mman.h>

//#include "dot-product-routines.h"
#include "sum-routines.h"



// Function prototypes

// Returns ns accurate walltime
double GetWallTime();

// Return mean of array
double ComputeMean(double * array, int arraySize);
// Return stdev of array
double ComputeStdev(double * array, int arraySize);
// Return max of array
double ComputeMax(double * array, int arraySize);

// Sets array[i] = i;
void InitializeArray(double * array, const int arraySize);

// Run different tests using function pointers. Argument "arg" not
// used by all routines, used for eg, for prefetch distance.
typedef double (*DotProductPtr)(CONST double * RESTRICT arrayA,
                                CONST double * RESTRICT arrayB,
                                CONST int arraySize,
                                CONST int arg);

// Run bandwidth test using supplied function. Argument "arg" passed
// to function to be tested.
void RunTest(DotProductPtr DotProduct, CONST int arraySize,
                                       CONST int nTimes,
                                       char * testName,
                                       CONST int arg);
void RunTestLP(DotProductPtr DotProduct, CONST int arraySize,
                                         CONST int nTimes,
                                         char * testName,
                                         CONST int arg);

// For array sum tests:
typedef double (*SumPtr)(CONST double * RESTRICT array,
                         CONST int arraySize,
                         CONST int arg);

// Run bandwidth test using supplied function. Argument "arg" passed
// to function to be tested.
void RunTestSum(SumPtr Sum, CONST int arraySize,
                            CONST int nTimes,
                            char * testName,
                            CONST int arg);
// Same, but allocates memory with large pages
void RunTestSumLP(SumPtr Sum, CONST int arraySize,
                              CONST int nTimes,
                              char * testName,
                              CONST int arg);



int main(void)
{
	char name[20];



	// Run Sum test without prefetch
	RunTestSum(&Sum1, ARRAYSIZE, NTIMES, "Sum1", 0);
	printf("\n");
	RunTestSum(&Sum2p2, ARRAYSIZE, NTIMES, "Sum2p2", 0);
	RunTestSum(&Sum2p4, ARRAYSIZE, NTIMES, "Sum2p4", 0);
	RunTestSum(&Sum2p8, ARRAYSIZE, NTIMES, "Sum2p8", 0);
	RunTestSum(&Sum2p16, ARRAYSIZE, NTIMES, "Sum2p16", 0);
	printf("\n");
	RunTestSum(&Sum3u1, ARRAYSIZE, NTIMES, "Sum3u1", 0);
	RunTestSum(&Sum3u2, ARRAYSIZE, NTIMES, "Sum3u2", 0);
	RunTestSum(&Sum3u4, ARRAYSIZE, NTIMES, "Sum3u4", 0);
	printf("\n");
	RunTestSum(&Sum4u1, ARRAYSIZE, NTIMES, "Sum4u1", 0);
	RunTestSum(&Sum4u2, ARRAYSIZE, NTIMES, "Sum4u2", 0);
	RunTestSum(&Sum4u4, ARRAYSIZE, NTIMES, "Sum4u4", 0);
	printf("\n");
	RunTestSum(&Sum52, ARRAYSIZE, NTIMES, "Sum52", 0);
	RunTestSum(&Sum54, ARRAYSIZE, NTIMES, "Sum54", 0);
	RunTestSum(&Sum58, ARRAYSIZE, NTIMES, "Sum58", 0);
	RunTestSum(&Sum516, ARRAYSIZE, NTIMES, "Sum516", 0);
	printf("\n");
	printf("\n");


#ifdef TESTHUGETLB
	// Run Sum test without prefetch, with hugetlb
	RunTestSum(&Sum1, ARRAYSIZE, NTIMES, "LP Sum1", 0);
	printf("\n");
	RunTestSumLP(&Sum2p2, ARRAYSIZE, NTIMES, "LP Sum2p2", 0);
	RunTestSumLP(&Sum2p4, ARRAYSIZE, NTIMES, "LP Sum2p4", 0);
	RunTestSumLP(&Sum2p8, ARRAYSIZE, NTIMES, "LP Sum2p8", 0);
	RunTestSumLP(&Sum2p16, ARRAYSIZE, NTIMES, "LP Sum2p16", 0);
	printf("\n");
	RunTestSumLP(&Sum3u1, ARRAYSIZE, NTIMES, "LP Sum3u1", 0);
	RunTestSumLP(&Sum3u2, ARRAYSIZE, NTIMES, "LP Sum3u2", 0);
	RunTestSumLP(&Sum3u4, ARRAYSIZE, NTIMES, "LP Sum3u4", 0);
	printf("\n");
	RunTestSumLP(&Sum4u1, ARRAYSIZE, NTIMES, "LP Sum4u1", 0);
	RunTestSumLP(&Sum4u2, ARRAYSIZE, NTIMES, "LP Sum4u2", 0);
	RunTestSumLP(&Sum4u4, ARRAYSIZE, NTIMES, "LP Sum4u4", 0);
	printf("\n");
	RunTestSumLP(&Sum52, ARRAYSIZE, NTIMES, "LP Sum52", 0);
	RunTestSumLP(&Sum54, ARRAYSIZE, NTIMES, "LP Sum54", 0);
	RunTestSumLP(&Sum58, ARRAYSIZE, NTIMES, "LP Sum58", 0);
	RunTestSumLP(&Sum516, ARRAYSIZE, NTIMES, "LP Sum516", 0);
	printf("\n");
	printf("\n");
#endif



// Run with HAND-TUNED optimal prefetch distances for each routine. (desktop values)
#ifdef SWPREFETCH
/*
	// Run Sum test with prefetch
	RunTestSum(&Sum1, ARRAYSIZE, NTIMES, "Sum1", 20*8);
	printf("\n");
	RunTestSum(&Sum2p2, ARRAYSIZE, NTIMES, "Sum2p2", 20*8);
	RunTestSum(&Sum2p4, ARRAYSIZE, NTIMES, "Sum2p4", 20*8);
	RunTestSum(&Sum2p8, ARRAYSIZE, NTIMES, "Sum2p8", 20*8);
	RunTestSum(&Sum2p16, ARRAYSIZE, NTIMES, "Sum2p16", 20*8);
	printf("\n");
	RunTestSum(&Sum3u1, ARRAYSIZE, NTIMES, "Sum3u1", 14*8);
	RunTestSum(&Sum3u2, ARRAYSIZE, NTIMES, "Sum3u2", 26*8);
	RunTestSum(&Sum3u4, ARRAYSIZE, NTIMES, "Sum3u4", 14*8);
	printf("\n");
	RunTestSum(&Sum4u1, ARRAYSIZE, NTIMES, "Sum4u1", 23*8);
	RunTestSum(&Sum4u2, ARRAYSIZE, NTIMES, "Sum4u2", 26*8);
	RunTestSum(&Sum4u4, ARRAYSIZE, NTIMES, "Sum4u4", 25*8);
	printf("\n");
	RunTestSum(&Sum52, ARRAYSIZE, NTIMES, "Sum52", 4*8);
	RunTestSum(&Sum54, ARRAYSIZE, NTIMES, "Sum54", 4*8);
	RunTestSum(&Sum58, ARRAYSIZE, NTIMES, "Sum58", 6*8);
	RunTestSum(&Sum516, ARRAYSIZE, NTIMES, "Sum516", 4*8);
	printf("\n");
	printf("\n");
*/

#ifdef TESTHUGETLB
/*
	// Run Sum test with prefetch, with hugetlb
	RunTestSum(&Sum1, ARRAYSIZE, NTIMES, "LP Sum1", 20*8);
	printf("\n");
	RunTestSumLP(&Sum2p2, ARRAYSIZE, NTIMES, "LP Sum2p2", 20*8);
	RunTestSumLP(&Sum2p4, ARRAYSIZE, NTIMES, "LP Sum2p4", 20*8);
	RunTestSumLP(&Sum2p8, ARRAYSIZE, NTIMES, "LP Sum2p8", 20*8);
	RunTestSumLP(&Sum2p16, ARRAYSIZE, NTIMES, "LP Sum2p16", 20*8);
	printf("\n");
	RunTestSumLP(&Sum3u1, ARRAYSIZE, NTIMES, "LP Sum3u1", 14*8);
	RunTestSumLP(&Sum3u2, ARRAYSIZE, NTIMES, "LP Sum3u2", 26*8);
	RunTestSumLP(&Sum3u4, ARRAYSIZE, NTIMES, "LP Sum3u4", 14*8);
	printf("\n");
	RunTestSumLP(&Sum4u1, ARRAYSIZE, NTIMES, "LP Sum4u1", 22*8);
	RunTestSumLP(&Sum4u2, ARRAYSIZE, NTIMES, "LP Sum4u2", 28*8);
	RunTestSumLP(&Sum4u4, ARRAYSIZE, NTIMES, "LP Sum4u4", 25*8);
	printf("\n");
	RunTestSumLP(&Sum52, ARRAYSIZE, NTIMES, "LP Sum52", 5*8);
	RunTestSumLP(&Sum54, ARRAYSIZE, NTIMES, "LP Sum54", 5*8);
	RunTestSumLP(&Sum58, ARRAYSIZE, NTIMES, "LP Sum58", 7*8);
	RunTestSumLP(&Sum516, ARRAYSIZE, NTIMES, "LP Sum516", 4*8);
	printf("\n");
	printf("\n");
*/
#endif
#endif


/*
#ifdef SWPREFETCH
	// Loop over prefetch distances, to determine best distance for each routine
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum1 %4d ", dist); RunTestSum(&Sum1, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum2p2 %4d ", dist); RunTestSum(&Sum2p2, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum2p4 %4d ", dist); RunTestSum(&Sum2p4, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum2p8 %4d ", dist); RunTestSum(&Sum2p8, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum2p16 %4d ", dist); RunTestSum(&Sum2p16, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum3u1 %4d ", dist); RunTestSum(&Sum3u1, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum3u2 %4d ", dist); RunTestSum(&Sum3u2, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum3u4 %4d ", dist); RunTestSum(&Sum3u4, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum4u1 %4d ", dist); RunTestSum(&Sum4u1, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum4u2 %4d ", dist); RunTestSum(&Sum4u2, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum4u4 %4d ", dist); RunTestSum(&Sum4u4, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum52 %4d ", dist); RunTestSum(&Sum52, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum54 %4d ", dist); RunTestSum(&Sum54, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum58 %4d ", dist); RunTestSum(&Sum58, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "Sum516 %4d ", dist); RunTestSum(&Sum516, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");

#ifdef TESTHUGETLB
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum1 %4d ", dist); RunTestSumLP(&Sum1, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum2p2 %4d ", dist); RunTestSumLP(&Sum2p2, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum2p4 %4d ", dist); RunTestSumLP(&Sum2p4, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum2p8 %4d ", dist); RunTestSumLP(&Sum2p8, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum2p16 %4d ", dist); RunTestSumLP(&Sum2p16, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum3u1 %4d ", dist); RunTestSumLP(&Sum3u1, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum3u2 %4d ", dist); RunTestSumLP(&Sum3u2, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum3u4 %4d ", dist); RunTestSumLP(&Sum3u4, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum4u1 %4d ", dist); RunTestSumLP(&Sum4u1, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum4u2 %4d ", dist); RunTestSumLP(&Sum4u2, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum4u4 %4d ", dist); RunTestSumLP(&Sum4u4, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum52 %4d ", dist); RunTestSumLP(&Sum52, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum54 %4d ", dist); RunTestSumLP(&Sum54, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum58 %4d ", dist); RunTestSumLP(&Sum58, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
	for (int dist = 0; dist < 64; dist++) {sprintf(name, "LPSum516 %4d ", dist); RunTestSumLP(&Sum516, ARRAYSIZE, NTIMES, name, dist*8);}
	printf("\n");
#endif
#endif
*/


	return 0;
}



void RunTest(DotProductPtr DotProduct, CONST int arraySize,
                                       CONST int nTimes,
                                       char * testName,
                                       CONST int arg)
{
	// Allocate (32B aligned, for AVX) arrays for test:
	double *arrayA, *arrayB;
	arrayA = _mm_malloc(ARRAYSIZE * sizeof *arrayA, 32);
	arrayB = _mm_malloc(ARRAYSIZE * sizeof *arrayB, 32);

	// Array to store bandwidth measurements
	double *bandwidth;
	bandwidth = malloc(nTimes * sizeof *bandwidth);
	// Array to store gflops measurements
	double *gflops;
	gflops = malloc(nTimes * sizeof *gflops);

	// store result of dot product, to check for errors
	double result = 0.0;

	// Initialize arrays
	InitializeArray(arrayA, arraySize);
	InitializeArray(arrayB, arraySize);

	for (int n = 0; n < nTimes; n++) {

		// Start timing
		double time = GetWallTime();

		result = DotProduct(arrayA, arrayB, arraySize, arg);

		// Stop timing
		time = GetWallTime() - time;

		// Compute test bandwidth
		bandwidth[n] = 2.0*(sizeof *arrayA)*(double)arraySize/pow(1024.0,3)/time;
		// Compute test gflops, for each element we do one mul and one add
		gflops[n] = (double)arraySize*(1.0 + 1.0)/1.0e9/time;
	}

	// Compute and print results
	double bandwidthMean = ComputeMean(bandwidth, nTimes);
	double bandwidthStdev = ComputeStdev(bandwidth, nTimes);
	double bandwidthMax = ComputeMax(bandwidth, nTimes);
	double gflopsMean = ComputeMean(gflops, nTimes);
	double relErr = (result-CORRECTRESULT)/CORRECTRESULT;
	double percOfPeak = bandwidthMean/PEAKBW*100.0;
	printf("%12s: %6.3lf GiB/s (stdev: %6.3lf ) (Max: %6.3lf GiB/s) (RelErr: %5.3le) ( %5.2lf %% peak) (%4.2lf GFLOPS)\n",
          testName, bandwidthMean, bandwidthStdev, bandwidthMax, relErr, percOfPeak, gflopsMean);

	// Free dynamic allocations
	_mm_free(arrayA);
	_mm_free(arrayB);
	free(bandwidth);
	free(gflops);
}



void RunTestLP(DotProductPtr DotProduct, CONST int arraySize,
                                         CONST int nTimes,
                                         char * testName,
                                         CONST int arg)
{
	// Allocate memory on huge pages. Need to request a multiple of 2MiB
	double *arrayA, *arrayB;
	int totalSize = (2*1024*1024)*(int)ceil((double)arraySize*(double)(sizeof *arrayA)/(2.0*1024.0*1024.0));
	// use mmap to reserve hugepages memory
	arrayA = mmap(NULL, totalSize, PROT_READ | PROT_WRITE , MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0);
	arrayB = mmap(NULL, totalSize, PROT_READ | PROT_WRITE , MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0);
	

	// Array to store bandwidth measurements
	double *bandwidth;
	bandwidth = malloc(nTimes * sizeof *bandwidth);
	// Array to store gflops measurements
	double *gflops;
	gflops = malloc(nTimes * sizeof *gflops);

	// store result of dot product, to check for errors
	double result = 0.0;

	// Initialize arrays
	InitializeArray(arrayA, arraySize);
	InitializeArray(arrayB, arraySize);

	for (int n = 0; n < nTimes; n++) {

		// Start timing
		double time = GetWallTime();

		result = DotProduct(arrayA, arrayB, arraySize, arg);

		// Stop timing
		time = GetWallTime() - time;

		// Compute test bandwidth
		bandwidth[n] = 2.0*(sizeof *arrayA)*(double)arraySize/pow(1024.0,3)/time;
		// Compute test gflops, for each element we do one mul and one add
		gflops[n] = (double)arraySize*(1.0 + 1.0)/1.0e9/time;
	}

	// Compute and print results
	double bandwidthMean = ComputeMean(bandwidth, nTimes);
	double bandwidthStdev = ComputeStdev(bandwidth, nTimes);
	double bandwidthMax = ComputeMax(bandwidth, nTimes);
	double gflopsMean = ComputeMean(gflops, nTimes);
	double relErr = (result-CORRECTRESULT)/CORRECTRESULT;
	double percOfPeak = bandwidthMean/PEAKBW*100.0;
	printf("%12s: %6.3lf GiB/s (stdev: %6.3lf ) (Max: %6.3lf GiB/s) (RelErr: %5.3le) ( %5.2lf %% peak) (%4.2lf GFLOPS)\n",
          testName, bandwidthMean, bandwidthStdev, bandwidthMax, relErr, percOfPeak, gflopsMean);

	// Free dynamic allocations
	munmap(arrayA, totalSize);
	munmap(arrayB, totalSize);
	free(bandwidth);
	free(gflops);
}



void RunTestSum(SumPtr Sum, CONST int arraySize,
                            CONST int nTimes,
                            char * testName,
                            CONST int arg)
{
	// Allocate (32B aligned, for AVX) arrays for test:
	double *array;
	array = _mm_malloc(ARRAYSIZE * sizeof *array, 32);

	// Array to store bandwidth measurements
	double *bandwidth;
	bandwidth = malloc(nTimes * sizeof *bandwidth);
	// Array to store gflops measurements
	double *gflops;
	gflops = malloc(nTimes * sizeof *gflops);

	// store result of sum, to check for errors
	double result = 0.0;

	// Initialize array
	InitializeArray(array, arraySize);

	for (int n = 0; n < nTimes; n++) {

		// Start timing
		double time = GetWallTime();

		result = Sum(array, arraySize, arg);

		// Stop timing
		time = GetWallTime() - time;

		// Compute test bandwidth
		bandwidth[n] = (sizeof *array)*(double)arraySize/pow(1024.0,3)/time;
		// Compute test gflops, for each element we do one add
		gflops[n] = (double)arraySize*(1.0)/1.0e9/time;
	}

	// Compute and print results
	double bandwidthMean = ComputeMean(bandwidth, nTimes);
	double bandwidthStdev = ComputeStdev(bandwidth, nTimes);
	double bandwidthMax = ComputeMax(bandwidth, nTimes);
	double gflopsMean = ComputeMean(gflops, nTimes);
	double relErr = (result-CORRECTSUM)/CORRECTSUM;
	double percOfPeak = bandwidthMean/PEAKBW*100.0;
	printf("%12s: %6.3lf GiB/s (stdev: %6.3lf ) (Max: %6.3lf GiB/s) (RelErr: %5.3le) ( %5.2lf %% peak) (%4.2lf GFLOPS)\n",
          testName, bandwidthMean, bandwidthStdev, bandwidthMax, relErr, percOfPeak, gflopsMean);

	// Free dynamic allocations
	_mm_free(array);
	free(bandwidth);
	free(gflops);
}



void RunTestSumLP(SumPtr Sum, CONST int arraySize,
                            CONST int nTimes,
                            char * testName,
                            CONST int arg)
{
	// Allocate memory on huge pages. Need to request a multiple of 2MiB
	double *array;
	int totalSize = (2*1024*1024)*(int)ceil((double)arraySize*(double)(sizeof *array)/(2.0*1024.0*1024.0));
	// use mmap to reserve hugepages memory
	array = mmap(NULL, totalSize, PROT_READ | PROT_WRITE , MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE | MAP_HUGETLB, -1, 0);

	// Array to store bandwidth measurements
	double *bandwidth;
	bandwidth = malloc(nTimes * sizeof *bandwidth);
	// Array to store gflops measurements
	double *gflops;
	gflops = malloc(nTimes * sizeof *gflops);

	// store result of sum, to check for errors
	double result = 0.0;

	// Initialize array
	InitializeArray(array, arraySize);

	for (int n = 0; n < nTimes; n++) {

		// Start timing
		double time = GetWallTime();

		result = Sum(array, arraySize, arg);

		// Stop timing
		time = GetWallTime() - time;

		// Compute test bandwidth
		bandwidth[n] = (sizeof *array)*(double)arraySize/pow(1024.0,3)/time;
		// Compute test gflops, for each element we do one add
		gflops[n] = (double)arraySize*(1.0)/1.0e9/time;
	}

	// Compute and print results
	double bandwidthMean = ComputeMean(bandwidth, nTimes);
	double bandwidthStdev = ComputeStdev(bandwidth, nTimes);
	double bandwidthMax = ComputeMax(bandwidth, nTimes);
	double gflopsMean = ComputeMean(gflops, nTimes);
	double relErr = (result-CORRECTSUM)/CORRECTSUM;
	double percOfPeak = bandwidthMean/PEAKBW*100.0;
	printf("%12s: %6.3lf GiB/s (stdev: %6.3lf ) (Max: %6.3lf GiB/s) (RelErr: %5.3le) ( %5.2lf %% peak) (%4.2lf GFLOPS)\n",
          testName, bandwidthMean, bandwidthStdev, bandwidthMax, relErr, percOfPeak, gflopsMean);

	// Free dynamic allocations
	munmap(array, totalSize);
	free(bandwidth);
	free(gflops);
}



void InitializeArray(double * array, const int arraySize)
{
	// Initialize in parallel (if we are using openmp). This puts the allocation
	// in the correct NUMA node on NUMA systems.
	#pragma omp parallel for default(none) shared(array)
	for (int i = 0; i < arraySize; i++) {
		array[i] = (double)i;
	}
}



double ComputeMean(double * array, int arraySize)
{
	double sum = 0.0;

	for (int i = 0; i < arraySize; i++) {
		sum += array[i];
	}

	return sum / (double)arraySize;
}



double ComputeStdev(double * array, int arraySize)
{
	double mean = ComputeMean(array, arraySize);
	double stdev = 0.0;

	for (int i = 0; i < arraySize; i++) {
		stdev += (array[i] - mean)*(array[i] - mean);
	}

	return sqrt(stdev/(double)(arraySize-1));
}



double ComputeMax(double * array, int arraySize)
{
	double max = DBL_MIN;

	for (int i = 0; i < arraySize; i++) {
		max = array[i] > max? array[i] : max;
	}

	return max;
}



double GetWallTime(void)
{
	struct timespec tv;
	clock_gettime(CLOCK_REALTIME, &tv);
	return (double)tv.tv_sec + 1e-9*(double)tv.tv_nsec;
}

