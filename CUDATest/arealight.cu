#include "arealight.h"

// Initializes white area light with given shape (cannot be NULL)
__device__ AreaLight::AreaLight(Shape* shape) {
	this->shape = shape;
	c = Color(1.f, 1.f, 1.f);
	i = 1.f;
}

// Initializes area light with given shape (cannot be NULL)
__device__ AreaLight::AreaLight(Shape* shape, Color color, float intensity, Point* loc) : Light(loc) {
	this->shape = shape;
	c = color;
	i = intensity;
}

// Returns true when this light intersects ray
// If so, output parameter t becomes the distance along ray to the closest intersection
__device__ bool AreaLight::Intersect(const Ray& ray, float& t) const {
	return shape->Intersect(ray, t);
}

__device__ Shape* AreaLight::GetShape() const {
	return shape;
}