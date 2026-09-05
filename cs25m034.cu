#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

#define CEIL_DIV(n, b) (((n) + (b) - (int)1) / (b))



__global__ void transpose(int* d_matrixA, int* d_matrixAT, int columns, int rows){

	//Dividing the matrix into 2d small blocks of size 32*32 
	//Launch configuration <<(ciel(columns/32),ciel(rows/32),1),(32,32,1)>>

	__shared__ int smallblockA[32][33]; // 33 to avoid bank comflicts

	// x is mapped to columns and y is mapped to rows

	unsigned idx = blockIdx.x*32 + threadIdx.x;
	unsigned idy = blockIdx.y*32 + threadIdx.y;

	if(idx < columns && idy < rows){
		smallblockA[threadIdx.y][threadIdx.x] = d_matrixA[idy*columns + idx]; // coalesced read
	}

	__syncthreads(); // making sure the smallblock is ready

	//calculating block position in the transposed matrix
	unsigned idtx = blockIdx.y * 32 + threadIdx.x; 
	unsigned idty = blockIdx.x * 32 + threadIdx.y;

	if(idtx < rows && idty < columns){
		d_matrixAT[idty*rows+idtx] = smallblockA[threadIdx.x][threadIdx.y]; // coalesces write
	}


}
__global__ void calculate(int* d_matrixAT, int* d_matrixB, int* d_matrixC, int* d_matrixDT, int* d_matrixE, int p, int q, int r){
	//launch configuration is <<<p,r>>>
	//ith block will only access elments of ith row elements in AT and C. So we can push them into shared memory
	__shared__ int ATi[1024];
	__shared__ int Ci[1024];
	unsigned i = blockIdx.x;
	//coalesced access pattern
	for(int k = threadIdx.x; k < q; k += r){
		ATi[k] = d_matrixAT[i * q + k];
		Ci[k] = d_matrixC[i*q+k];
	}

	__syncthreads(); // making sure the rows are fully loaded
	unsigned j = threadIdx.x;
	int ele = 0;
	// Access to BT and D are coalesced and AT and C are in shared memory;
	for(int k = 0; k < q; k++){
		ele += ATi[k]* d_matrixB[k*r+j];
		ele += Ci[k]*d_matrixDT[k*r+j];
	}

	d_matrixE[i*r+j] = ele; // Access to E is coalesced

}

// function to compute the output matrix
void compute(int p, int q, int r, int *h_matrixA, int *h_matrixB,
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE,*d_matrixAT,*d_matrixDT;

	// allocate memory...
	cudaMalloc(&d_matrixA, q * p * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, q * p * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */
	cudaMalloc(&d_matrixAT, q * p * sizeof(int));
	cudaMalloc(&d_matrixDT, r * q * sizeof(int));

	int a = CEIL_DIV(p,32);
    int b = CEIL_DIV(q,32);
    dim3 grid(a,b,1);
    dim3 block(32,32,1);
	transpose<<<grid,block>>>(d_matrixA,d_matrixAT,p,q);
	int a1 = CEIL_DIV(q,32);
    int b1 = CEIL_DIV(r,32);
    dim3 grid1(a1,b1,1);
    dim3 block1(32,32,1);
	transpose<<<grid1,block1>>>(d_matrixD,d_matrixDT,q,r);
	calculate<<<p,r>>>(d_matrixAT,d_matrixB,d_matrixC,d_matrixDT,d_matrixE,p,q,r);


	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
	cudaFree(d_matrixAT);
	cudaFree(d_matrixDT);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}



int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;

    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n");
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(q * p * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, q, p);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	compute(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
