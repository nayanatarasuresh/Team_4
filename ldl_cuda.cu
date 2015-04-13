//libraries
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
// or #include "cuda_runtime.h"
// #include <cuda_runtime_api.h>
//#include <curand_kernel.h>
#include <cusolverDn.h>
//#include <cusolver.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <cublas_v2.h>
#include <sys/time.h>
#include "team4_readme.h"

/* Macros */
#define DATASIZE     15 //1-15

double timer_val() {
		struct timeval st;
		gettimeofday( &st, NULL );
		return (st.tv_sec+st.tv_usec*1e-6);
	} 
	
int main()
{
	int i, j;
	int data;
	int n;
	int lda = 0;
	clock_t start = 0, end = 0;
	double exec_time;
	cusolverStatus_t status_ldl; 
	cusolverStatus_t buff_size;
	FILE *fp = fopen("output_ldl_cuda.csv", "w"); 
	fclose(fp);
	
	// Running the LDLT for matrices of different sizes
	for (data = 1; data <= DATASIZE; data++) {
		n = exp2((double)data);
		
		double *A = (double*)malloc(n*n*sizeof(double));
	
		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++) {
				
				if ( i == j) {
					*((A + i) + n*j) = i + ( (j + 1) * (3 + i));
				}
				else {
					*((A + i) + n*j) = i + j ;
				}
				
			}
			
		}
		
		
		
		//allocating memory on the GPU by copying the matrix here
		double *Z;
		cudaMalloc(&Z, n*n*sizeof(double));		
		cudaMemcpy(Z, A, n*n*sizeof(double), cudaMemcpyHostToDevice);
		
		//enabling the handle
		cusolverDnHandle_t handle;	
    	cusolverDnCreate(&handle);
		
		// 
		int *devInfo; 
		cudaMalloc(&devInfo, sizeof(int)); // ask if this needs allocation of memory
		
		lda = n;
		int sizeof_work = 0;
		buff_size = cusolverDnDsytrf_bufferSize(handle, n, Z, lda, &sizeof_work);	
		
		double *work;
		cudaMalloc(&work, sizeof_work * sizeof(double));
		int *ipiv;
		cudaMalloc(&ipiv, n*sizeof(int));
		
		start = timer_val();
		//start = cusolver_test_seconds();
		for(j=0; j<1000; j++)	
		{
			//cusolverDnDsytrf(cusolverDnHandle_t handle,cublasFillMode_t uplo,int n,double *A,int lda,int *ipiv,double *work,int lwork,int *devInfo );
		
			// Computing the LDLT decomposition
			status_ldl = cusolverDnDsytrf(handle, CUBLAS_FILL_MODE_LOWER, n, Z, lda, ipiv, work, sizeof_work, devInfo);
			
			printf ("\n The device info is %d\n", devInfo);
			
			if ( status_ldl == CUSOLVER_STATUS_SUCCESS ) {
				printf ("\nThe status of the LDLT decomposition is CUSOLVER_STATUS_SUCCESS");
			}
			else if ( status_ldl == CUSOLVER_STATUS_NOT_INITIALIZED ) {
				printf ("\nThe status of the LDLT decomposition is CUSOLVER_STATUS_NOT_INITIALIZED");
			}
			else if ( status_ldl == CUSOLVER_STATUS_INVALID_VALUE ) {
				printf ("\nThe status of the LDLT decomposition is CUSOLVER_STATUS_INVALID_VALUE");
			}
			else if ( status_ldl == CUSOLVER_STATUS_ARCH_MISMATCH ) {
				printf ("\nThe status of the LDLT decomposition is CUSOLVER_STATUS_ARCH_MISMATCH");
			}
			else if ( status_ldl == CUSOLVER_STATUS_INTERNAL_ERROR ) {
				printf ("\nThe status of the LDLT decomposition is CUSOLVER_STATUS_INTERNAL_ERROR");
			}
		} 
		end = timer_val();
		//end = cusolver_test_seconds();
		
		exec_time = (double)(end - start);
		//exec_time = exec_time;                         //dividing by 1000 because we have executed 1000 iterations
		
		printf ("The execution for LDLT decomposition of a %d by %d matrix is %.15f s\n", n, n, exec_time/1000.00);
		//printf ("The execution for LDLT decomposition of a %d by %d matrix is %.8f s\n", n, n, exec_time*1000);
			
		// open file where the output should be saved
		FILE *fp = fopen("output_ldl_cuda.csv", "a"); 
		fprintf(fp, "The datasize is %d\n", n);
		fprintf(fp, "The execution for LDLT decomposition of a %d by %d matrix is %.15f s\n", n, n, exec_time/1000.00);
		fclose(fp);
		
		free(A);
		cusolverDnDestroy(handle);
		cudaFree(Z);
	}
}	


/*things to ask
ask if devInfo needs allocation of memory and the cmds to exxecute cuda 
cusolver_test_seconds not working
*/
