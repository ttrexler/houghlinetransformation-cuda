//random
//2
//test
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define iceil(num, den) (num + den - 1) / den
#define ARRAY_SIZE 1000
#define BIN 25

//kernel functions
__global__ void kernelHough(int *d_array, int size) {
	__shared__ int chunk[BIN];//number of bins
	/*
	take a piece of the array. discretize into y=mx+b format per point. check all points and increment all bins touched
	at the end recombine all shared memory to a global bin tally. Take the most significant X numbers as lines.
	discretized from point(1,1) :: (from format y=mx+b) y=x+1 
	check each bin 
	*/
}

//prep function
void houghTransform(int* h_input_array, int size){
	int *d_array;
	int asize = size * sizeof(int);
	cudaMalloc((void**)&h_input_array, asize);
	cudaMemcpy(d_array, h_input_array, asize, cudaMemcpyHostToDevice);
}

int main() {
	//test case array
	int *h_test_input = new int[20];
	int test[20] = {1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10}; //{x1,y1,x2,y2...}
	//random array initializer
	int *random=new int[ARRAY_SIZE];
	srand(time(0));
	for (int i = 0; i < ARRAY_SIZE; i++) {
		random[i] = rand() % 10;
	}
	//begin test function
	houghTransform(test,20);
	return 0;
}
