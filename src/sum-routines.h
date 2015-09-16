// Function prototypes

#define RESTRICT restrict
#define CONST const

#define SWPREFETCH "1"

double Sum1(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);

double Sum2p2(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);
double Sum2p4(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);
double Sum2p8(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);
double Sum2p16(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);

double Sum3u1(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);
double Sum3u2(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);
double Sum3u4(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);

double Sum4u1(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);
double Sum4u2(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);
double Sum4u4(CONST double * RESTRICT array,
            CONST int arraySize,
            CONST int prefetchDistance);
