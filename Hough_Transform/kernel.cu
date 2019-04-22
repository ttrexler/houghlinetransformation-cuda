#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define iceil(num, den) (num + den - 1) / den
#define ARRAY_SIZE 20 //must be an even number; this number/2 = number of points //sets random array and constant mem size
//#define BIN 100 //divides the grid into square bins to vote on. perfect square value
#define NUM_LINES 1 //top X voted lines. Picks first X Largest from top left to bottom right of grid space.

/*GRID evaluated for bin voting
 * Must always be a square grid with origin at center
 */
#define LXBOUND -5 //lowest X
#define RXBOUND 5 //highest X
#define LYBOUND -5 //lowest Y
#define UYBOUND 5 //highest Y
////////////////////////////////

#define INCREMENT 0.25 //precision, length of 1 side of the square(bin)
//The (abs)difference between between two sides is the length of the grid. Length/Increment determines how many bins 

__constant__ int d_coordarray[ARRAY_SIZE];//Place coordinates in constant memory

//show grid with votes. Becomes unuseful when bins > 20x20
void printVotes(int *h_binarray) {
	// Number of columns
	const int col = (((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT)) / ((RXBOUND + UYBOUND) / INCREMENT);

	for (int i = 0; i < col; ++i) {
		for (int j = 0; j < col * col; j += col)
			std::cout << h_binarray[i + j] << "\t";
		std::cout << std::endl;
	}
}

// Convert from array index to representative slope
float slopeCalculator(int index) {
	const int col = ((((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT)) / ((RXBOUND + UYBOUND) / INCREMENT));
	const int center = ((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT) / 2;

	int displacement = 0, flag = 0;
	int change = col;

	while (flag == 0) {
		if (index <= center + change && index >= center - change) {
			flag++;
		} else {
			change += col;
			displacement++;
		}
	}

	return (displacement * INCREMENT) + (INCREMENT / 2.0);
}

// Convert from array index to representative intercept
float interceptCalculator(int index) {
	const int col = ((((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT)) / ((RXBOUND + UYBOUND) / INCREMENT));
	const int check = index % col;

	int displacement = 0, flag = 0;
	int center1 = col / 2, center2 = col / 2 - 1;

	while (flag == 0) {
		((check == center1 || check == center2) ? flag : displacement)++;
		center1++;
		center2--;
	}

	return (displacement * INCREMENT) + (INCREMENT / 2.0);
}

// Find n highest indexes in the array
void highest_index(int *h_binarray) {
	const int size = ((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT);
	const int col = (((RXBOUND - LXBOUND)*(RXBOUND - LXBOUND)) / ((RXBOUND + UYBOUND) * INCREMENT));

	int *index = new int[size];
	for (int i = 0; i < size; ++i)
		index[i] = i;

	bool stop = true;

	int temp, temp2;

	// Bubble sort
	for (int i = 1; (i <= size) && stop; ++i) {
		stop = false;

		for (int j = 0; j < (size - 1); ++j) {
			if (h_binarray[j + 1] > h_binarray[j]) {
				temp = h_binarray[j];
				temp2 = index[j];

				h_binarray[j] = h_binarray[j + 1];
				index[j] = index[j + 1];

				h_binarray[j + 1] = temp;
				index[j + 1] = temp2;

				stop = true;
			}
		}
	}

	//use highest values for slope & intercept
	float totalslope = 0.0, totalintercept = 0.0;

	for (int i = 0; i < NUM_LINES; ++i) {
		const float slope = slopeCalculator(index[i]);
		const float intercept = interceptCalculator(index[i]);

		std::cout << "[" << i << "]: ";

		if (index[i] < (size / 2)) {
			std::cout << "slope= -" << slope << " and " << std::endl;

			totalslope = totalslope - slope;
		} else {
			cout << "slope = " << slope << " and " std::endl;

			totalslope = totalslope + slope;
		}

		if (index[i] % col < (col / 2)) {
			std::cout	<< " and intercept = " << intercept << std::endl
						<< "From point: " << index[i] << std::endl;

			totalintercept = totalintercept + intercept;
		} else {
			std::cout	<< "and intercept = -" << intercept << std::endl
						<< "From point: " << index[i] << std::endl;

			totalintercept = totalintercept - intercept;
		}
	}

	std::cout << "=============" << std::endl;
	std::cout << "The average of these slopes is :" << totalslope / NUM_LINES;
	std::cout << "The average of these intercept is:" << totalintercept / NUM_LINES;
	std::cout << std::endl;
}

//kernel functions
__global__ void kernelHough(int size, int* d_binarray) {
	//__shared__ int chunk[BIN];//number of bins
	/*
	take a piece of the array. discretize into y=mx+b format per point. check all points and increment all bins touched
	at the end recombine all shared memory to a global bin tally. Take the most significant X numbers as lines.
	discretized from point(1,1) ==(m,n)==> (-1,1)
	check each bin for count and sum them to a global array in sync
	NUM of coordinates will check all bins for their own equation and increment appropriately
	*/

	// Number from 0 through arraysize / 2
	const int thread = 2 * (blockDim.x * blockIdx.x + threadIdx.x);

	// Slope is discretized space = -1 / x
	const float slope = -1 * (1 / d_coordarray[thread]);

	// Intercept in discretized space = y / x
	const float intercept = d_coordarray[thread + 1] / d_coordarray[thread];

	int counter = 0;
	for (float x = LXBOUND; x < RXBOUND; x += INCREMENT) {
		for (float y = UYBOUND; y > LYBOUND; y -= INCREMENT) {
			const float xMin = x;
			const float xMax = x + INCREMENT;
			
			const float yMin = y - INCREMENT;
			const float yMax = y;
			
			const float lower_range = slope * xMin + intercept;
			const float upper_range = slope * xMax + intercept;

			if ((lower_range <= yMax && lower_range >= yMin) || (upper_range <= yMax && upper_range >= yMin))
				atomicAdd(&d_binarray[counter], 1);

			counter++;
		}
	}
}

//prep function
void houghTransform(int* h_input_array, int size) {
	int *d_binarray;
	int *h_binarray = new int[((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT)];

	// Length of the square grid for bins * size of int
	const int binarraysize = (((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT)) * sizeof(int);
	const int coordarraysize = size * sizeof(int);

	// Copy coordinates to Constant Memory
	cudaMemcpyToSymbol(d_coordarray, h_input_array, coordarraysize);
	cudaMalloc((void**)&d_binarray, binarraysize);

	// 1-D Block
	dim3 myBlockDim(1, 1, 1);

	// ((size / 2), 1, 1); 1d grid
	dim3 myGridDim((size/2), 1, 1);

	kernelHough <<<myGridDim, myBlockDim>>> (size, d_binarray);

	cudaMemcpy(h_binarray, d_binarray, binarraysize, cudaMemcpyDeviceToHost);

	printVotes(h_binarray);
	highest_index(h_binarray);
}

int main() {
	// Seed RNG
	srand(time(0));

	// Test case array
	int *h_test_input = new int[20];

	int test[20] = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10};

	// Random array initializer
	int *random = new int[ARRAY_SIZE];
	for (int i = 0; i < ARRAY_SIZE; ++i)
		random[i] = (rand() % 10) + 1;

	// Begin test function
	houghTransform(test, 20);

	return 0;
}