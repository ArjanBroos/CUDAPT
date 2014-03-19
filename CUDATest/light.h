#pragma once

#include "color.h"
#include "geometry.h"

// The abstract base class for all lights
class Light {
public:
	Color			c;	// Color of light
	float			i;	// Intensity of light

	// Returns true when this light intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ virtual bool	Intersect(const Ray& ray, float& t) const = 0;

	// Returns the emitted light
	__device__ Color		Le() const;
};