#include "light.h"

//Constructor
__device__ Light::Light() {};
//Constructor to set the location of the light
__device__ Light::Light(Point* loc) : Object(loc) {}

// Returns the emitted light
__device__ Color Light::Le() const {
	return i * c;
}