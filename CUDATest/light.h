#ifndef LIGHT_H
#define LIGHT_H

#include "color.h"
#include "geometry.h"
#include "object.h"
#include "shape.h"

// The abstract base class for all lights
class Light: public Object {
public:
	__device__ Light();

	Color			c;	// Color of light
	float			i;	// Intensity of light

	// Returns true when this light intersects ray
	// If so, output parameter t becomes the distance along ray to the closest intersection
	__device__ virtual bool	Intersect(const Ray& ray, float& t) const = 0;
	// Returns the type of this object
	__device__ ObjectType GetType() const;
	// Return the corner point of object
	__device__ virtual const Point*		GetCornerPoint() const = 0;

	// Returns the emitted light
	__device__ Color		Le() const;
};

#endif
