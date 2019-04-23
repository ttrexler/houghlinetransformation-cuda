# Hough Line Transformation (Using CUDA) - C++

---

## Code breakdown

---

### #**define** explanations
#### ARRAY_SIZE
    Must be even. This  size divided by two is the number of coordinates passed to all subsequent functions that utilize this information and thus defines how many lines are evaluated in the discretized space. 

#### NUM_LINES
    Specifies what number of highest voted bin values you would like to see the slope and intercepts of.
    
#### RXBOUND / LXBOUND / UYBOUND / LYBOUND
    Define the area about the origin which will be divided into bins and voted upon to determine the most probable fit line to the data set.

#### INCREMENT
    Determines the dimension of the bins on the graph. Larger numbers will be less precise due to overlapping lines in the transformed space. The number of calculations increase exponentially as you lower the increment size. E.G. in an area bound by x:-5,5 y:-5,5 if the increment is 1 you will have 100 bins. However, if the increment is 0.01 you will have 1 million bins. 
    
#### COLUMN
    Is a frequently used variable defined by:
    (((RXBOUND - LXBOUND) / INCREMENT) * ((RXBOUND - LXBOUND) / INCREMENT)) / ((RXBOUND + UYBOUND) / INCREMENT)
    which amounts determining the number of columns in the bin space. 

---

### Function explanations
#### main()
    Input Array of random coordinates on the (x,y) plane is initialized and passed to the kernel preparation function "houghTransform()"
    
#### houghTransform(int *h_array, int size)
    This function prepares the device/host array allocations, copying of data between host and device, and setup of the grid and block structure. Once all allocations are made this function calls the kernel function "kernelHough()" and then "highestIndex()"
    
#### kernelHough(int size, int *d_binarray)
    First identifies the threadid which is an index for a single point in the xy plane. Calculates slope in discretized space (-1/x) Calculates Intercept in discretized space (y/x) Calculates the lower and upper bounds for each bin in the graph by finding the xMin and xMax values per bin. I then multiply this value by slope and add intercept value to find the respective lower and upper bounds. Finally if the lower or the upper bounds fall between the yMin and yMax, atomicAdd increments that bin’s counter and moves on to the next bin until every bin is evaluated per coordinate.
    
#### highestIndex(int *h_binarray)
    Initializes a new array of size equivalent to the bin array in which each value is set to its index(index[0] = 0, index[1] = 1, ..., index[n] = n). Implements a bubble sort on the values in the bin array and concurrently moves the indices to match. The result is an array in descending order of the highest voted bin indices. Calls the functions "slopeCalculator" and "interceptCalculator". The highest NUMLINES values slope and intercepts are the calculated and reported. 
    
#### slopeCalculator(int index)
    Passed the index of one of the highest voted bins. Determines how many columns from the center the index lies and stores the value as "displacement". (displacement * INCREMENT) + (INCREMENT / 2) will give the value of the slope at the center of the bin.

#### interceptCalculator(int index)
    Passed the index of one of the highest voted bins. Determines how many rows from the center the index lies and stores the value as "displacement". 
    (displacement * INCREMENT) + (INCREMENT / 2) will give the value of the intercept at the center of the bin.

## Note:
**The values returned as a result of this method can only be within INCREMENT / 2 precision with the true answer since the center of the bins are used as the representative points.
E.G. if the true slope is 1 and you set the INCREMENT=0.01, the most precise result to be expected would be 1.005.**

**It is also important to realize that the highest voted bins are generally clumped together  and result in a degree of skew in precision. Also, due to the way the bins are calculated and then sorted the bins chosen to represent the highest votes ( in the event of multiple highest voted bins) is chosen from top left to bottom right if one were looking at a graph representing 4 quadrants.**

---

### Installation
This program was compiled using [Visual Studio Community 2017](https://visualstudio.microsoft.com/downloads/) with [CUDA Tookit 10.1](https://developer.nvidia.com/cuda-downloads) for Windows 10 x64.

---

# Contributors
[ttrexler](https://github.com/ttrexler)
[AWildTeddyBear](https://github.com/AWildTeddyBear)
