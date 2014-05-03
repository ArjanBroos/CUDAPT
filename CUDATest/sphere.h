#ifndef SPHERE_H
#define SPHERE_H

#include "geometry.h"
#include "shape.h"

// Represents a sphere
class Sphere : public Shape {
public:
	Point p;	// Lower Left Front point of BB
	Point c;	// Center
	float r;	// Radius

	// Initializes a sphere at (0, 0, 0) with radius 1
	__device__ Sphere();
	// Initializes a sphere at c with radius r
	__device__ Sphere(const Point& c, float r);
	// Initializes a sphere at c with radius r
	__device__ Sphere(const Point& c);

	// Returns true when this sphere intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ bool	Intersect(const Ray& ray, float& t) const;

	// Returns the normal of this sphere at point p
	__device__ Vector	GetNormal(const Point& p) const;

	// Returns the type of this shape
	__device__ ShapeType GetType() const;

	// Return the corner point of object
	__device__ const Point*		GetCornerPoint() const;
};

#endif
