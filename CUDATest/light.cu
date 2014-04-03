#include "light.h"

//Constructor
__device__ Light::Light() {};

// Returns the emitted light
__device__ Color Light::Le() const {
	return i * c;
}

__device__ ObjectType Light::GetType() const {
	return OBJ_LIGHT;
}