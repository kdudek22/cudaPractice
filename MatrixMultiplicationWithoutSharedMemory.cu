#include <cuda_runtime.h>
#include <stdio.h>
#define TILE_SIZE 2

__global__ void addMatrix(float* a, float* b, float* c, int matrixSize) {
	int row = blockIdx.x * TILE_SIZE + threadIdx.x;
	int col = blockIdx.y * TILE_SIZE + threadIdx.y;


	if (row < matrixSize && col < matrixSize) {
		int res = 0;

		for (int i = 0; i < matrixSize; i++) {
			res += a[row * matrixSize + i] * b[i * matrixSize + col];
		}

		c[row * matrixSize + col] = res;
	}
}

void printMatrix(float* matrix, int matrixDim) {
	for (int i = 0; i < matrixDim; i++) {
		for (int j = 0; j < matrixDim; j++) {
			printf("%f ", matrix[i * matrixDim + j]);
		}
		printf("\n");
	}
}

int main() {
	int matrixDim = 10;

	float* matrixA = new float[matrixDim * matrixDim];

	float* matrixB = new float[matrixDim * matrixDim];

	float* matrixC = new float[matrixDim * matrixDim];

	// wypelnienie wartosciami
	for (int i = 0; i < matrixDim; i++) {
		for (int j = 0; j < matrixDim; j++) {
			matrixA[i * matrixDim + j] = 1;
			matrixB[i * matrixDim + j] = 1;
		}
	}

	float* d_a, * d_b, * d_c;

	cudaMalloc(&d_a, matrixDim * matrixDim * sizeof(float));
	cudaMalloc(&d_b, matrixDim * matrixDim * sizeof(float));
	cudaMalloc(&d_c, matrixDim * matrixDim * sizeof(float));

	cudaMemcpy(d_a, matrixA, matrixDim * matrixDim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, matrixB, matrixDim * matrixDim * sizeof(float), cudaMemcpyHostToDevice);


	dim3 gridDim ((matrixDim - 1)/TILE_SIZE + 1, (matrixDim - 1)/TILE_SIZE + 1);
	dim3 blockDim (TILE_SIZE, TILE_SIZE);


	addMatrix <<<gridDim, blockDim >>> (d_a, d_b, d_c, matrixDim);

	cudaDeviceSynchronize();

	cudaMemcpy(matrixC, d_c, matrixDim * matrixDim * sizeof(float), cudaMemcpyDeviceToHost);


	printMatrix(matrixA, matrixDim);
	printf("========\n");
	printMatrix(matrixB, matrixDim);
	printf("========\n");
	printMatrix(matrixC, matrixDim);


	return 0;
}
