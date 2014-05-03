#ifndef PLANE_H
#define PLANE_H

#include "shape.h"
#include "geometry.h"

// Represents an infinite plane
class Plane : public Shape {
public:
	Point	p;	// A point on the plane
	Vector	n;	// Normal of the plane

	// Initializes a plane through (0, 0, 0) with normal (0, 1, 0)
	__device__ Plane();
	// Initializes a plane through p with normal n
	__device__ Plane(const Point& p, const Vector& n);

	// Returns true when this shape intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ bool		Intersect(const Ray& ray, float& t) const;
	
	// Returns the normal of this shape at point p
	__device__ Vector	GetNormal(const Point& p) const;

	// Returns the type of this shape
	__device__ ShapeType GetType() const;

	// Return the corner point of object
	__device__ virtual const Point*		GetCornerPoint() const;
};

#endif
