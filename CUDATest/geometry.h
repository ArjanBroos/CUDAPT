#ifndef GEOMETRY_H
#define GEOMETRY_H

#include "point.h"
#include "vector.h"
#include "ray.h"
#include "cuda_inc.h"

// Returns the dot product of two vectors
__host__ __device__ inline float Dot(const Vector& v1, const Vector& v2) {
	return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

// Returns the cross product of two vectors
__host__ __device__ inline Vector Cross(const Vector& v1, const Vector& v2) {
	return Vector(
		v1.y*v2.z - v1.z*v2.y,
		v1.z*v2.x - v1.x*v2.z,
		v1.x*v2.y - v1.y*v2.x);
}

// Returns the unit vector with the same direction as v
__host__ __device__ inline Vector Normalize(const Vector& v) {
	float factor = 1.f / v.Length();
	return v * factor;
}

#endif
