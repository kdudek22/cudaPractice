#include <cuda_runtime.h>
#include <stdio.h>

__global__ void addMatrix() {

	printf("%d-%d-%d, ", blockDim.x, blockIdx.x, threadIdx.x);
}


int main() {
	
	int nCols = 3;
	int nRows = 3;


	// We create a array of pointers to 
	float ** x = new float* [nRows];

	// each value in the array is a float[]
	for (int i = 0; i < nRows; i++) {
		x[i] = new float[nCols];
	}


	// this fills the array with values
	for (int i=0; i < nRows; i++) {
		for (int j = 0; j < nCols; j++) {
			x[i][j] = float(i*10) + float(j);
		}
	}

	// Tu specyfikujesz ile chcesz odpalic bloków i ile watków w bloku
	// przykladowo jezeli odpalimy <<<2,10>>>, to:
	// blockDim.x = 10, ilosc watkow w bloku
	// blockIdx.x - indeksy blokow, my odpalilismy 2 wiec od {0,1}
	// threadIdx.x - id danego watku w bloku, my w kazdym bloku uruchomilismy 10 watkow, wiec wartosci {0..9}
	addMatrix <<<3, 100 >> >();


	cudaDeviceSynchronize();


	// print the array
	for (int i=0; i < nRows; i++) {
		for (int j=0; j < nCols; j++) {
			printf("%f, ", x[i][j]);
		}
		printf("\n");
		
	}

	// free up the allocated memory
	for (int i = 0; i < nRows; i++) {
		delete[] x[i];
	}

	delete[]x;

	return 0;
}
