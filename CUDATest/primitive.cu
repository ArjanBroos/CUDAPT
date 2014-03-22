#include "primitive.h"

// Initializes this primitive with a shape and a material
__device__ Primitive::Primitive(Shape* shape, Material* material, Point* loc) : Object(loc), shape(shape), material(material) {
}

// Returns a pointer to the shape
__device__ const Shape* Primitive::GetShape() const {
	return shape;
}

// Returns the material
__device__ const Material* Primitive::GetMaterial() const {
	return material;
}

// Checks if the shape of the primitive intersects with a ray and store the distance in t
__device__ bool Primitive::Intersect(const Ray& ray, float& t) const {
	return shape->Intersect(ray, t);
}