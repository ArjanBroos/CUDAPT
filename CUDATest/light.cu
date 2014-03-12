#include "light.h"

// Returns the emitted light
__device__ Color Light::Le() const {
	return i * c;
}