#ifndef OBJECT_H
#define OBJECT_H

#include "point.h"
#include "ray.h"

enum ObjectType{
	OBJ_LIGHT = 0, 
	OBJ_PRIMITIVE,
};

class Node;
class Object
{
public:
    Object() {};

	//Checks if the ray intersects with the shape of the object
    virtual bool	Intersect(const Ray& ray, float& t) const = 0;
	// Returns the type of this object
    virtual ObjectType GetType() const = 0;
    Node* GetParent();
    void SetParent(Node* parent);
	// Return the corner point of object
    virtual const Point*		GetCornerPoint() const = 0;

private:
	Node*		parent;
};

#endif
