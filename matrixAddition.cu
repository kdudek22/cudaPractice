#include <cuda_runtime.h>
#include <stdio.h>
#define TPB 64

__global__ void addMatrix(float *a, float *b, float *c, int matrixSize) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	//printf("%d - %d, ", index, matrixSize);
	
	if (index < matrixSize) {
		c[index] = a[index] + b[index];
	}
}

void printMatrix(float* matrix, int nCols, int nRows) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			printf("%f ", matrix[i*nCols + j]);
		}
		printf("\n");
	}
}

int main() {
	int nCols = 15;
	int nRows = 3;

	float* matrixA = new float[nRows*nCols];

	float* matrixB = new float[nRows*nCols];

	float* matrixC = new float[nRows*nCols];
	
	// wypelnienie wartosciami
	for (int i=0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			matrixA[i * nCols + j] = float(i + j);
			matrixB[i * nCols + j] = float(i + j);
		}
	}

	float *d_a, * d_b, * d_c;
	
	cudaMalloc(&d_a, nCols * nRows * sizeof(float));
	cudaMalloc(&d_b, nCols * nRows * sizeof(float));
	cudaMalloc(&d_c, nCols * nRows * sizeof(float));

	cudaMemcpy(d_a, matrixA, nCols * nRows * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, matrixB, nCols * nRows * sizeof(float), cudaMemcpyHostToDevice);

	int blockDim = (nCols * nRows - 1) / TPB + 1;
	printf("Block dimensions: %d\n", blockDim);

	addMatrix<<<blockDim, TPB >> > (d_a, d_b, d_c, nCols * nRows);

	cudaDeviceSynchronize();

	cudaMemcpy(matrixC, d_c, nCols * nRows * sizeof(float), cudaMemcpyDeviceToHost);


	printMatrix(matrixA, nCols, nRows);
	printf("========\n");
	printMatrix(matrixB, nCols, nRows);
	printf("========\n");
	printMatrix(matrixC, nCols, nRows);


	return 0;
}
