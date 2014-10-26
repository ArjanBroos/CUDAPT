#ifndef PRIMITIVE_H
#define PRIMITIVE_H

#include "shape.h"
#include "material.h"
#include "object.h"

// Represents a physical object with a shape and material
class Primitive: public Object {
public:
	// Default constructor
    Primitive() {};
	// Initializes this primitive with a shape and a material
    Primitive(Shape* shape, Material* material);

	// Returns a pointer to the shape
    const Shape*	GetShape() const;
	// Returns the material
    const Material*	GetMaterial() const;
	// Checks if the shape of the primitive intersects with a ray and store the distance in t
    bool	Intersect(const Ray& ray, float& t) const;
	// Returns the type of this object
    ObjectType GetType() const;
	// Return the corner point of object
    virtual const Point*		GetCornerPoint() const;

private:
	Shape*			shape;		// Shape of the object
	Material*		material;	// Material of the object
};

#endif
