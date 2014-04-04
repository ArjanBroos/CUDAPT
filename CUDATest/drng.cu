#include "drng.h"

// randomValues should be a pointer to already generated and allocated random values between 0 and 1
// width and height correspond to the dimensions of the screen
// nrPerPixel is the number of random values
__device__ DRNG::DRNG(float* randomValues, unsigned width, unsigned height, unsigned n) {
	this->randomValues = randomValues;
	this->width = width;
	this->height = height;
	this->n = n;

	// Initialize indices to 0
	unsigned resolution = width*height;
	indices = new unsigned[resolution];
	for (unsigned i = 0; i < resolution; i++)
		indices[i] = (i * 91) % n;
}

// Return the next random value for the pixel at (x, y)
__device__ float DRNG::Next(unsigned x, unsigned y) {
	const unsigned pixelIndex = y * width + x;
	const unsigned valueIndex = indices[pixelIndex]++;
	const float value = randomValues[valueIndex];

	if (valueIndex >= n)
		indices[pixelIndex] = 0;

	return value;
}