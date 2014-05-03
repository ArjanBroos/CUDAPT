#ifndef COLOR_H
#define COLOR_H

#include "cuda_inc.h"

class Color {
public:
	float r, g, b;	// Color components

	__host__ __device__ Color();
	__host__ __device__ Color(float r, float g, float b);

	__host__ __device__ Color operator+(const Color& c) const;
	__host__ __device__ Color& operator+=(const Color& c);
	__host__ __device__ Color operator*(const Color& c) const;
	__host__ __device__ Color& operator*=(const Color& c);
	__host__ __device__ Color operator*(float s) const;
	__host__ __device__ Color& operator*=(float s);
	__host__ __device__ Color operator/(float s) const;
	__host__ __device__ Color& operator/=(float s);
};

__host__ __device__ inline Color operator*(float s, const Color& c) {
	return Color(c.r * s, c.g * s, c.b * s);
}

#endif
