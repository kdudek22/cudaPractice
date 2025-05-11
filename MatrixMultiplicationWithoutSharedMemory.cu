#include <cuda_runtime.h>
#include <stdio.h>
#define TILE_SIZE 2

__global__ void addMatrix(float* a, float* b, float* c, int a_rows, int a_cols, int b_rows, int b_cols) {
	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;


	if (row < a_rows && col < b_cols) {
		int res = 0;

		for (int i = 0; i < a_cols; i++) {
			res += a[row * a_cols + i] * b[i * b_cols + col];
		}

		c[row * b_cols + col] = res;
	}
}

void printMatrix(float* matrix, int rows, int cols) {
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%f ", matrix[i * cols + j]);
		}
		printf("\n");
	}
}

void fillMatrix(float* matrix, int n_rows, int n_cols) {
	for (int i = 0; i < n_rows; i++) {
		for (int j = 0; j < n_cols; j++) {
			matrix[i * n_cols + j] = 1;
		}
	}
}

int main() {
	int a_rows = 3, a_cols = 1;
	int b_rows = 1, b_cols = 3;

	float* matrixA = new float[a_rows * a_cols];

	float* matrixB = new float[b_rows * b_cols];

	float* matrixC = new float[a_rows * b_cols];

	if (a_rows != b_cols) {
		printf("Could not multiply matrixes, as the dimensions do not match");
		return 0;
	}

	fillMatrix(matrixA, a_rows, a_cols);
	fillMatrix(matrixB, b_rows, b_cols);

	float* d_a, * d_b, * d_c;

	cudaMalloc(&d_a, a_rows * a_cols * sizeof(float));
	cudaMalloc(&d_b, b_rows * b_cols * sizeof(float));
	cudaMalloc(&d_c, a_rows * b_cols * sizeof(float));

	cudaMemcpy(d_a, matrixA, a_rows * a_cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, matrixB, b_rows * b_cols * sizeof(float), cudaMemcpyHostToDevice);


	dim3 gridDim ((b_cols - 1)/TILE_SIZE + 1, (a_rows - 1)/TILE_SIZE + 1);
	dim3 blockDim (TILE_SIZE, TILE_SIZE);


	addMatrix <<<gridDim, blockDim >>> (d_a, d_b, d_c, a_rows, a_cols, b_rows, b_cols);

	cudaDeviceSynchronize();

	cudaMemcpy(matrixC, d_c, a_rows * b_cols * sizeof(float), cudaMemcpyDeviceToHost);


	printMatrix(matrixA, a_rows, a_cols);
	printf("========\n");
	printMatrix(matrixB, b_rows, b_cols);
	printf("========\n");
	printMatrix(matrixC, a_rows, b_cols);


	return 0;
}
