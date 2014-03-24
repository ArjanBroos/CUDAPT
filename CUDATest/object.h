#ifndef OBJECT_H
#define OBJECT_H

//#include "point.h"
//#include "ray.h"
//#include "octree.h"

enum objectType{LIGHT, PRIMITIVE, PLANE};

class Point;
class Node;
class Ray;
class Object
{
public:
	__device__ Object() {};
	__device__ Object(Point* loc) : loc(loc), size(-1) {};
	__device__ Object(int size, Point* loc) : size(size), loc(loc) {};
	int			size;
	Point*		loc;
	int			type;
	Node*		parent;

	//Checks if the ray intersects with the shape of the object
	__device__ virtual bool	Intersect(const Ray& ray, float& t) const = 0;
};

#endif