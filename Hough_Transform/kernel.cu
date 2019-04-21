#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

#define iceil(num, den) (num + den - 1) / den
#define ARRAY_SIZE 20 //must be an even number; this number/2 = number of points
#define BIN 100 //divides the grid into square bins to vote on. perfect square value
#define NUM_LINES 8 //top X voted lines
#define LXBOUND -5
#define RXBOUND 5
#define LYBOUND -5
#define UYBOUND 5
#define INCREMENT 1.0

__constant__ int d_coordarray[ARRAY_SIZE];//Place coordinates in constant memory

using namespace std;

//show grid with votes
void printVotes(int *h_binarray) {
	//int size = (RXBOUND - LXBOUND)*(RXBOUND - LXBOUND);//array index size
	int col = ((RXBOUND - LXBOUND)*(RXBOUND - LXBOUND)) / (RXBOUND + UYBOUND);//number of columns
	for (int i = 0; i < col; i++) {
		for (int j = 0; j < col*col; j += col) {
			cout << h_binarray[i + j] << "    ";
		}
		cout <<endl;
	}

}
float slopeCalculator(int index) {//convert from array index to representative slope
	int col = (((RXBOUND - LXBOUND)*(RXBOUND - LXBOUND)) / (RXBOUND + UYBOUND));
	int change = col;
	int center = (RXBOUND - LXBOUND)*(RXBOUND - LXBOUND)/2;
	int displacement = 0;
	float returnval;
	int  flag = 0;
	while (flag == 0) {
		if (index<=center + change && index>=center - change) {
			flag++;
		}
		else {
			change +=col ; displacement++;
		}
	}
	returnval= (displacement*INCREMENT) + (INCREMENT / 2.0);
	return returnval;
}

float interceptCalculator(int index) {//convert from array index to representative intercept
	int col = (((RXBOUND - LXBOUND)*(RXBOUND - LXBOUND)) / (RXBOUND + UYBOUND));
	int displacement = 0;
	int check= index%col;
	int center1 = col / 2;
	int center2 = col / 2 - 1;
	float returnval;
	int flag = 0;
	while (flag == 0) {
		if (check == center1 || check == center2) {
			flag++;
		}
		else displacement++;
		center1++; center2--;
	}
	returnval = (displacement*INCREMENT) + (INCREMENT / 2.0);
	return returnval;
}

//find n highest indexes in the array
void highest_index(int *h_binarray) {
	int size = (RXBOUND - LXBOUND)*(RXBOUND - LXBOUND);
	int *index=new int[size];
	for (int i = 0; i < size; i++) { index[i] = i; }//array representing indices
	int stop = 1;// 1 starts, then 0, end on 1
	int temp,temp2;             
	//bubble sort
	for (int i = 1; (i <= size) && stop ; i++)
	{
		stop = 0;
		for (int j = 0; j < (size - 1); j++)
		{
			if (h_binarray[j + 1] > h_binarray[j])    
			{
				temp = h_binarray[j];  
				temp2 = index[j];
				h_binarray[j] = h_binarray[j + 1];
				index[j] = index[j + 1];
				h_binarray[j + 1] = temp;
				index[j+1] = temp2;
				stop = 1;
			}
		}
	}
	//use highest values for slope & intercept
	int col = (((RXBOUND - LXBOUND)*(RXBOUND - LXBOUND)) / (RXBOUND + UYBOUND));
	float totalslope=0.0, totalintercept=0.0;
	for (int i = 0; i < NUM_LINES; i++) {
		float slope = slopeCalculator(index[i]);
		float intercept = interceptCalculator(index[i]);
		cout << "Line " << i<<":";
		if (index[i] < (size / 2)) {
			cout << "slope= -" << slope << " and \n";
			totalslope = totalslope - slope;
		}
		else {
			cout << "slope=" << slope << " and \n";
			totalslope = totalslope + slope;
		}
		if (index[i] % col < (col / 2)) {
			cout << " and intercept =" << intercept << endl;
			cout << "From point:" << index[i] << endl;
			totalintercept = totalintercept + intercept;
		}
		else {
			cout << "and intercept = -" << intercept << endl;
			cout << "From point: " << index[i] << endl;
			totalintercept = totalintercept - intercept;
		}
	}
	cout << "=============\n";
	cout << "The average of these slopes is :" << totalslope / NUM_LINES;
	cout << "The average of these intercept is:" << totalintercept / NUM_LINES;
	cout << endl;
	

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
	int thread = 2 * (blockDim.x * blockIdx.x + threadIdx.x);//number from 0 through arraysize/2 
	float slope = -1 * (1 / d_coordarray[thread]); // slope is discretized space = -1/x
	float intercept = d_coordarray[thread + 1] / d_coordarray[thread]; // intercept in discretized space = y/x
	int counter = 0;
	for (float x = LXBOUND; x < RXBOUND; x += INCREMENT) {
		for (float y = UYBOUND; y > LYBOUND; y -= INCREMENT) {
			float xMin = x;
			float xMax = x + INCREMENT;
			float yMin = y - INCREMENT;
			float yMax = y;
			float lower_range = slope * xMin + intercept;
			float upper_range = slope * xMax + intercept;
			if ((lower_range<=yMax && lower_range>=yMin) || (upper_range<=yMax && upper_range>=yMin)) {
				atomicAdd(&d_binarray[counter], 1);
			}
			counter++;
		}
	}
}

//prep function
void houghTransform(int* h_input_array, int size) {
	int *d_binarray;
	int *h_binarray = new int[(RXBOUND - LXBOUND)*(RXBOUND - LXBOUND)];
	int coordarraysize = size * sizeof(int);
	int binarraysize = (RXBOUND - LXBOUND)*(RXBOUND - LXBOUND) * sizeof(int); // length of the square grid for bins * size of int
	cudaMemcpyToSymbol(d_coordarray, h_input_array, coordarraysize);//copy coordinates to Constant Memory
	cudaMalloc((void**)&d_binarray, binarraysize);
	dim3 myBlockDim(1, 1, 1);//1d block
	dim3 myGridDim((size/2), 1, 1);//((size / 2), 1, 1);//1d grid
	kernelHough << <myGridDim, myBlockDim >> > (size, d_binarray);
	cudaMemcpy(h_binarray, d_binarray, binarraysize, cudaMemcpyDeviceToHost);
	printVotes(h_binarray);
	highest_index(h_binarray);
}

int main() {
	//test case array
	cout << "begin\n";
	int *h_test_input = new int[20];
	int test[20] = { 1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10 }; //{x1,y1,x2,y2...}
	//random array initializer
	int *random = new int[ARRAY_SIZE];
	srand(time(0));
	for (int i = 0; i < ARRAY_SIZE; i++) {
		random[i] = rand() % 10;
	}
	//begin test function
	houghTransform(test, 20);
	return 0;
}