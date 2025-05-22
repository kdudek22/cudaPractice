#include <cuda_runtime.h>
#include <stdio.h>
#define TPB 1024


__global__ void reduce0(float * vector, float * res, int n_elements) {
	extern __shared__ float sdata[];

	int tid = threadIdx.x;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = vector[i];
	__syncthreads();

	for (int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid == 0) {
		res[blockIdx.x] = sdata[0];
	}
}

void fill_vector(float* vector, int vector_length) {
	for (int i = 0; i < vector_length; i++) {
		vector[i] = float(1);
	}
}

int main() {
	int n_elements = 2048;

	float* vector = new float[n_elements];
	float* res = new float[n_elements];

	fill_vector(vector, n_elements);

	int current_size = n_elements;

	float* d_vector, *d_res;

	cudaMalloc(&d_vector, n_elements * sizeof(float));
	cudaMalloc(&d_res, n_elements * sizeof(float));

	int blocks = (n_elements - 1) / TPB + 1;

	cudaMemcpy(d_vector, vector, n_elements * sizeof(float), cudaMemcpyHostToDevice);

	reduce0 <<<blocks, TPB, TPB * sizeof(float) >> > (d_vector, d_res, n_elements);

	cudaMemcpy(res, d_res, n_elements * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_vector);
	cudaFree(d_res);

	cudaDeviceSynchronize();

	printf("%f", res[0]);

	

	return 0;
}
