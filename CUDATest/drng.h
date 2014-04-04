#pragma once

#include "cuda_inc.h"

// Device random number generator
// Upload a lot of random numbers to the device and then use this class to handle them more easily
class DRNG {
public:
	// randomValues should be a pointer to already generated and allocated random values between 0 and 1
	// width and height correspond to the dimensions of the screen
	// n is the number of random values
	__device__ DRNG(float* randomValues, unsigned width, unsigned height, unsigned n);

	// Return the next random value for the pixel at (x, y)
	__device__ float	Next(unsigned x, unsigned y);

private:
	float*		randomValues;	// All the random values
	unsigned*	indices;		// Indices. At which random value are we for this pixel?
	unsigned	width;			// Number of pixels in a row
	unsigned	height;			// Number of pixels in a column
	unsigned	n;				// Number of random values
};