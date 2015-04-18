/*
icc -mmic -mkl ldl_mkl.c -o ldl_mkl -lm
scp ldl_mkl mic0:ldl_mkl

ssh mic0

./ldl_mkl


*/

#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <mkl_lapacke.h>
#include <time.h>
#include <math.h>
#include <sys/time.h>
#include "team4_readme.h"
#include <omp.h>


/* Macros */
#define DATASIZE     1 //1-15

int main()
{
	int i, j, k;
	int data;
	lapack_int n;
	clock_t start = 0, end = 0;
	double exec_time;
	srand(0);
	lapack_int return_val = 0;
	
	
	double timer_val() {
		struct timeval st;
		gettimeofday( &st, NULL );
		return (st.tv_sec+st.tv_usec*1e-6);
	} 
	
	FILE *fp = fopen("output_ldl.csv", "w"); 
	fclose(fp);
	
	// Running the LDLT for matrices of different sizes
	for (data = 1; data <= DATASIZE; data++) {
		n = exp2(data);
		double *A = (double*)malloc(n*n*sizeof(double));
		double *e = (double*)malloc(n*sizeof(double));
		double *d = (double*)malloc(n*sizeof(double));

		// Initializing the input matrix A, arrays d and e
		
		
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				
				if ( i == j) {
					*((A + i) + n*j) = i + ( (j + 1) * (3 + i));
					*(d + i) = *((A + i) + n*j);
				}
				else {
					*((A + i) + n*j) = 0;
					if ( j == i - 1 || j == i + 1) {
						*((A + i) + n*j) = i + j ;
						
					}
				}
				
			}
			
		}
		
		for (i = 0; i < n; i++) {
			for (j = 0, k=0; j < n, k < n-1; j++, k++) {
				if ( j == i - 1 ) {
					*(e + k) = *((A + i) + n*j);
				}
			}
		}
		
		/*printf (" the matrix a is \n");
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				printf ("%lf\t", *((A + i) + n*j)); 
			}
			printf ("\n");
		}  
		
		printf ("\n");
		for (i = 0; i < n-1; i++) {
			printf ("e %lf\t", *(e + i));
		} 
		
		
		printf ("\n");
		for (i = 0; i < n; i++) {
			printf ("d %lf\t", *(d + i));
		}
		*/
		return_val = 0;
		//start time
		start = clock();
		//start = timer_val();
		//start = omp_get_wtime();
		double *e_new = (double*)malloc(n*sizeof(double));
		double *d_new = (double*)malloc(n*sizeof(double));
		for ( k = 0; k < 1000; k++) {
			//lapack_int LAPACKE_dpttrf( lapack_int n, double* d, double* e );
			
			d_new = d;
			e_new = e;
			return_val = LAPACKE_dpttrf(n, d_new, e_new);
			
			//printf ("return value = %d\n", return_val);
			d = d_new;
			e = e_new;
					
			d_new = NULL;
			e_new = NULL;
		}
		
		/*
		printf ("\n AFter\n");
		printf ("e \n");
		for (i = 0; i < n-1; i++) {
			printf (" %f\t", *(e + i));
		} 
				
		printf ("\n d \n");
		for (i = 0; i < n; i++) {
			printf (" %f\t", *(d + i));
		}
		*/
		
		//stop time
		end = clock();
		//end = timer_val();
		//end = omp_get_wtime();
		exec_time = (double)(end - start) / CLOCKS_PER_SEC;
		//exec_time = (double)(end - start);
		//exec_time = exec_time;                         //dividing by 1000 because we have executed 1000 iterations
		
		//printf ("The execution for LDLT decomposition of a %d by %d matrix is %.15f s\n", n, n, exec_time/1000.00);
		printf ("The execution for LDLT decomposition of a %d by %d matrix is %.8f e-10s\n", n, n, exec_time*1000000000);
			
		// open file where the output should be saved
		FILE *fp = fopen("output_ldl.csv", "a"); 
		fprintf(fp, "The datasize is %d\n", n);
		fprintf(fp, "The execution for LDLT decomposition of a %d by %d matrix is %.15f s\n", n, n, exec_time/1000.00);
		fclose(fp);
		
		free(A);
		free(e);
		free(d);
		free(d_new);
		free(e_new);
	}

	return 0;
}
