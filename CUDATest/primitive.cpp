#include "primitive.h"

// Initializes this primitive with a shape and a material
Primitive::Primitive(Shape* shape, Material* material) : shape(shape), material(material) {
}

// Returns a pointer to the shape
const Shape* Primitive::GetShape() const {
	return shape;
}

// Returns the material
const Material* Primitive::GetMaterial() const {
	return material;
}

// Checks if the shape of the primitive intersects with a ray and store the distance in t
bool Primitive::Intersect(const Ray& ray, float& t) const {
	return shape->Intersect(ray, t);
}

// Returns the type of this object
ObjectType Primitive::GetType() const {
	return OBJ_PRIMITIVE;
}

// Return the corner point of object
const Point*		Primitive::GetCornerPoint() const {
	return shape->GetCornerPoint();
}
