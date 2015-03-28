#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_lapacke.h>
#include <time.h>
#include <math.h>
#include <team4_readme.h>

/* Macros */
#define DATASIZE     15 //1-15

int main()
{
	int i, j, k;
	int data;
	lapack_int n;
	clock_t start = 0, end = 0;
	double exec_time;
	srand(0);
	
	// Running the LDLT for matrices of different sizes
	for (data = 1; data < DATASIZE; data++) {
		n = exp2(data);
		double *A = (double*)malloc(n*n*sizeof(double));
		double *e = (double*)malloc(n*sizeof(double));
		double *d = (double*)malloc(n*sizeof(double));

		// Initializing the input matrix A, arrays d and e
		for (i = 1; i <= n; i++) {
			for (j = 1; j <= n; j++) {
				*((A + i) + j) = 0;
				*((A + i) + j) = ((double)rand() / (double)RAND_MAX);
				if (i == j) {
					*(d + i) = *((A + i) + j);
				} else if (i > j) {
					*(e + i) = *((A + i) + j);
				}
			}
		}
	
		//start time
		start = clock();
		for ( k = 0; k < 1000; k++) {
			//lapack_int LAPACKE_dpttrf( lapack_int n, double* d, double* e );
			LAPACKE_dpttrf(n, d, e);
		}
		//stop time
		end = clock();
		exec_time = ((double)(end - start)) / CLOCKS_PER_SEC;
		exec_time = exec_time/1000;                         //dividing by 1000 because we have executed 1000 iterations
			
		// open file where the output should be saved
		FILE *fp = fopen("output.cvs", "w"); 
		fprintf(fp, "The datasize is %d", n);
		fprintf(fp, "The execution for LDLT decomposition of a %d by %d matrix is %f\n", n, n, exec_time);
		fclose(fp);
	}

	return 0;
}
