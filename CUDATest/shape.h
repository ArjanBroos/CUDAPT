#pragma once

#include "geometry.h"

// The abstract base class for all shapes
class Shape {
public:
	// Returns true when this shape intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ virtual bool	Intersect(const Ray& ray, float& t) const = 0;
	
	// Returns the normal of this shape at point p
	__device__ virtual Vector	GetNormal(const Point& p) const = 0;
};