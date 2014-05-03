#ifndef SHAPE_H
#define SHAPE_H

#include "geometry.h"

enum ShapeType {
	ST_START = 0,
	ST_CUBE = ST_START,
	ST_SPHERE,
	ST_PLANE,
	ST_END = ST_PLANE,
};

// The abstract base class for all shapes
class Shape {
public:
	// Returns true when this shape intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ virtual bool	Intersect(const Ray& ray, float& t) const = 0;
	
	// Returns the normal of this shape at point p
	__device__ virtual Vector	GetNormal(const Point& p) const = 0;

	// Returns the type of this shape
	__device__ virtual ShapeType GetType() const = 0;

	// Return the corner point of object
	__device__ virtual const Point*		GetCornerPoint() const = 0;
};

#endif
