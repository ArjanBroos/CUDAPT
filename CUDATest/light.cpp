#include "light.h"

//Constructor
Light::Light() {};

// Returns the emitted light
Color Light::Le() const {
	return i * c;
}

ObjectType Light::GetType() const {
	return OBJ_LIGHT;
}
