#ifndef QUATERNION_H
#define QUATERNION_H

#include "vector.h"

class Quaternion {
public:
	__host__ __device__ Quaternion() {};
	__host__ __device__ Quaternion( Vector& n, float a);
	
	__host__ __device__ float Length() const;
	__host__ __device__ Quaternion Normalize() const;
	__host__ __device__ Quaternion Inverted() const;
	__host__ __device__ Quaternion operator*(const Quaternion& q) const;
	__host__ __device__ Vector operator*(const Vector& q) const;

	float w;
	Vector v;
};

#endif
