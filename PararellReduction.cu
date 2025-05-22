#include <cuda_runtime.h>
#include <stdio.h>

#define TPB 1024  // Threads per block

__global__ void reduce0(float* vector, float* res, int n_elements) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = vector[i];
    __syncthreads();

    // Perform reduction in shared memory (works since n_elements is a power of 2)
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2 * s)) == 0)  sdata[tid] += sdata[tid + s];

        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tid == 0) {
        res[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce1(float* vector, float* res, int n_elements) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = vector[i];
    __syncthreads();

    // Perform reduction in shared memory (works since n_elements is a power of 2)
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;

        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // Write the result of this block to global memory
    if (tid == 0) {
        res[blockIdx.x] = sdata[0];
    }
}

__global__ void reduce2(float* vector, float* res, int n_elements) {
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    sdata[tid] = vector[i];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    // Write the result of this block to global memory
    if (tid == 0) {
        res[blockIdx.x] = sdata[0];
    }
}



void fill_vector(float* vector, int vector_length) {
    for (int i = 0; i < vector_length; i++) {
        vector[i] = float(i);  // Simple test data: all ones
    }
}

int main() {
    int n_elements = 32768;  // Must be a power of 2

    float* vector = new float[n_elements];
    fill_vector(vector, n_elements);

    // Allocate host result buffer
    float* res = new float[1];  // Final result will be a single float

    // Allocate device memory
    float* d_vector, * d_res;
    cudaMalloc(&d_vector, n_elements * sizeof(float));
    cudaMalloc(&d_res, n_elements * sizeof(float));  // Over-allocated, reused during reduction

    // Copy input to device
    cudaMemcpy(d_vector, vector, n_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Perform reduction iteratively on GPU
    float* in = d_vector;
    float* out = d_res;
    int current_size = n_elements;

    while (current_size > 1) {
        int threads = (current_size < TPB) ? current_size : TPB;
        int blocks = current_size / threads;

        reduce1 << <blocks, threads, threads * sizeof(float) >> > (in, out, current_size);
        cudaDeviceSynchronize();  // Optional but helpful for debugging

        current_size = blocks;

        // Swap pointers
        float* tmp = in;
        in = out;
        out = tmp;
    }

    // Copy the final result from device to host
    cudaMemcpy(res, in, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Final sum: %f\n", res[0]);  // Expected: n_elements * 1.0 = 2048.0

    // Cleanup
    cudaFree(d_vector);
    cudaFree(d_res);
    delete[] vector;
    delete[] res;

    return 0;
}
