#pragma once

#include "cuda_inc.h"
#include <string>

// Represents a vector in 3-dimensional space
class Vector {
public:
	// Components of the vector
	float x, y, z;

	// Initialize vector as (0, 0, 0)
	__host__ __device__ Vector();
	// Initialize vector as (x, y, z)
	__host__ __device__ Vector(float x, float y, float z);

	// Returns the length of the vector
	__host__ __device__ float Length() const;
	// Returns the length of the vector squared - faster than Length()
	__host__ __device__ float LengthSquared() const;

	// Comparison operator
	__host__ __device__ bool operator==(const Vector& v) const;

	// Arithmetic operators
	__host__ __device__ Vector operator-() const;
	__host__ __device__ Vector operator+(const Vector& v) const;
	__host__ __device__ Vector& operator+=(const Vector& v);
	__host__ __device__ Vector operator*(float s) const;
	__host__ __device__ Vector& operator*=(float s);
	__host__ __device__ Vector operator/(float s) const;
	__host__ __device__ Vector& operator/=(float s);
	__host__ __device__ Vector operator-(const Vector& v) const;
	__host__ __device__ Vector& operator-=(const Vector& v);

	__host__ std::string ToString() const;
};

// Arithmetic operators
__host__ __device__ inline Vector operator*(float s, const Vector& v) {
	return Vector(s * v.x, s * v.y, s * v.z);
}