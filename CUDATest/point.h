#pragma once

#include "cuda_inc.h"
#include "vector.h"

// Represents a 3-dimensional point
class Point {
public:
	// Point coordinates
	float x, y, z;

	// Initialize point as (0, 0, 0)
	__host__ __device__ Point();
	// Initialize point as (x, y, z)
	__host__ __device__ Point(float x, float y, float z);

	// Arithmetic operators
	__host__ __device__ Vector operator-(const Point& p) const;
	__host__ __device__ Point operator+(const Vector& v) const;
	__host__ __device__ Point& operator+=(const Vector& v);
	__host__ __device__ Point operator-(const Vector& v) const;
	__host__ __device__ Point& operator-=(const Vector& v);
	__host__ __device__ Point operator+(const Point& p) const;
	__host__ __device__ Point& operator+=(const Point& p);
	__host__ __device__ Point operator*(float s) const;
	__host__ __device__ Point& operator*=(float s);
	__host__ __device__ Point operator/(float s) const;
	__host__ __device__ Point& operator/=(float s);
	__host__ __device__ bool operator>(const Point &p) const;
	__host__ __device__ bool operator<(const Point &p) const;
};

inline Point operator*(float s, const Point& p) {
	return Point(s * p.x, s * p.y, s* p.z);
}