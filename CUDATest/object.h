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
	__device__ Object() {};

	//Checks if the ray intersects with the shape of the object
	__device__ virtual bool	Intersect(const Ray& ray, float& t) const = 0;
	// Returns the type of this object
	__device__ virtual ObjectType GetType() const = 0;
	__device__ Node* GetParent();
	__device__ void SetParent(Node* parent);
	// Return the corner point of object
	__device__ virtual const Point*		GetCornerPoint() const = 0;

private:
	Node*		parent;
};

#endif