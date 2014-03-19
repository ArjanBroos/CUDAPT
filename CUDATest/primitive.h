#pragma once

#include "shape.h"
#include "material.h"

// Represents a physical object with a shape and material
class Primitive {
public:
	// Initializes this primitive with a shape and a material
	__device__ Primitive(Shape* shape, Material* material);

	// Returns a pointer to the shape
	__device__ const Shape*	GetShape() const;
	// Returns the material
	__device__ const Material*	GetMaterial() const;

private:
	Shape*			shape;		// Shape of the object
	Material*		material;	// Material of the object
};