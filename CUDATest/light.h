#pragma once

#include "color.h"
#include "geometry.h"
#include "object.h"
#include "shape.h"

// The abstract base class for all lights
class Light: public Object {
public:
	__device__ Light();
	__device__ Light(Point* loc);

	Color			c;	// Color of light
	float			i;	// Intensity of light

	// Returns true when this light intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ virtual bool	Intersect(const Ray& ray, float& t) const = 0;

	// Returns the emitted light
	__device__ Color		Le() const;
};