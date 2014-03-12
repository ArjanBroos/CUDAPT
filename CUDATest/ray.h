#pragma once

#include "point.h"
#include "vector.h"
#include "math.h"
#include "cuda_inc.h"

// Represents a ray with an origin and direction through 3-dimensional space
class Ray {
public:
	Point		o;	// Origin
	Vector		d;	// Direction

	// The ray is clamped between o + mint * d and o + maxt * d
	// To prevent intersection with a surface that this ray originated from
	float		mint;	// Minimum distance along ray for intersection checks
	float		maxt; // Maximum distance along ray for intersection checks

	// Initializes a ray with origin o and direction d. mint and maxt are optional
	__host__ __device__ Ray(const Point& o, const Vector& d, float mint = 1e-6f, float maxt = INFINITY);

	// Returns o + t*d
	__host__ __device__ Point		operator()(float t) const;
};