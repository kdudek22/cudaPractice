#include <cuda_runtime.h>
#include <stdio.h>

#define TPB 64
#define RADIUS 1

void printVector(float* vector, int length) {
	for (int i = 0; i < length; i++) {
		printf("%f, ", vector[i]);
	}
	printf("\n");
}

__global__ void filter(float* a, float* b, int length) {
	__shared__ float memory[TPB + 2 * RADIUS];
	int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int innerIndex = threadIdx.x + RADIUS;

	memory[innerIndex] = a[globalIndex];

	if (threadIdx.x < RADIUS) {
		memory[innerIndex - RADIUS] = a[globalIndex - RADIUS < 0 ? length + globalIndex - RADIUS : globalIndex - RADIUS];

		memory[innerIndex + TPB] = a[globalIndex + TPB >= length ? globalIndex + TPB - length : globalIndex + TPB];
	}

	__syncthreads();

	float result = 0.0;

	for (int offset = -RADIUS; offset <= RADIUS; offset++) {
		result += memory[innerIndex + offset];
	}
	printf("%d\n", innerIndex);
	b[globalIndex] = result;
}


int main() {
	int nElements = 5;

	float* vector = new float[nElements];
	float* res = new float[nElements];

	for (int i = 0; i < nElements; i++) {
		vector[i] = float(1);
	}

	float *d_vector, *d_res;
	
	cudaMalloc(&d_vector, nElements * sizeof(float));
	cudaMalloc(&d_res, nElements * sizeof(float));

	cudaMemcpy(d_vector, vector, nElements * sizeof(float), cudaMemcpyHostToDevice);

	int blockDim = (nElements - 1) / TPB + 1;
	int sharedMemoryAmount = (TPB + 2 * RADIUS) * sizeof(float);

	printf("%d, %d\n", blockDim, sharedMemoryAmount);

	filter<<<blockDim, TPB, sharedMemoryAmount>>>(d_vector, d_res, nElements);

	cudaMemcpy(res, d_res, nElements * sizeof(float), cudaMemcpyDeviceToHost);


	printVector(vector, nElements);
	printVector(res, nElements);

	return 0;
}
