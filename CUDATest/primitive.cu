#include "primitive.h"

// Initializes this primitive with a shape and a material
__device__ Primitive::Primitive(Shape* shape, Material* material) : shape(shape), material(material) {
}

// Returns a pointer to the shape
__device__ const Shape* Primitive::GetShape() const {
	return shape;
}

// Returns the material
__device__ const Material* Primitive::GetMaterial() const {
	return material;
}