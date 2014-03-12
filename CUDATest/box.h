#pragma once

#include "shape.h"

// Represents a sphere
class Box : public Shape {
public:
	//Upper and lower bound points of the box
	Point bounds[2];

	//Create a box of size 1 at (x,y,z)
	__device__ Box(Point& p);
	//Create a box of size s at (x,y,z)
	__device__ Box(Point& p, float s);
	//Create a box with lower bound p and upper bound q
	__device__ Box(Point& p, Point& q);

	// Returns true when this box intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ bool		Intersect(const Ray& ray, float& t) const;

	// Returns the normal of this box at point p
	__device__ Vector	GetNormal(const Point& p) const;
};