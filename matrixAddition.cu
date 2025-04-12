#include <cuda_runtime.h>
#include <stdio.h>
#define TPB 64

__global__ void addMatrix(float *a, float *b, float *c, int matrixSize) {
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	//printf("%d - %d, ", index, matrixSize);
	
	if (index < matrixSize) {
		c[index] = a[index] + b[index];
		printf("%d %d\n", index, a[index]);
	}
}

void printMatrix(float** matrix, int nCols, int nRows) {
	for (int i = 0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			printf("%f ", matrix[i][j]);
		}
		printf("\n");
	}
}

int main() {
	int nCols = 3;
	int nRows = 3;

	float** matrixA = new float*[nRows];
	matrixA[0] = new float[nRows * nCols];

	float** matrixB = new float* [nRows];
	matrixB[0] = new float[nRows * nCols];

	float** matrixC = new float* [nRows];
	matrixC[0] = new float[nRows * nCols];

	// cd alokacji pamieci
	for (int i = 1; i < nRows; i++) {
		matrixA[i] = matrixA[0] + i * nCols;
		matrixB[i] = matrixB[0] + i * nCols;
		matrixC[i] = matrixC[0] + i * nCols;
	}

	// wypelnienie macierzy wartosciami
	for (int i = 0;i < nRows;i++) {
		for (int j = 0; j < nCols; j++) {
			matrixA[i][j] = float(i*nCols + j);
			matrixB[i][j] = float(i*nCols + j);
		}
	}

	float *d_a, * d_b, * d_c;
	
	cudaMalloc(&d_a, nCols * nRows * sizeof(float));
	cudaMalloc(&d_b, nCols * nRows * sizeof(float));
	cudaMalloc(&d_c, nCols * nRows * sizeof(float));

	for (int i = 0; i < nRows; i++) {
		cudaMemcpy(&d_a[i * nCols], matrixA[i], sizeof(float) * nCols, cudaMemcpyHostToDevice);
		cudaMemcpy(&d_b[i * nCols], matrixB[i], sizeof(float) * nCols, cudaMemcpyHostToDevice);
	}

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
