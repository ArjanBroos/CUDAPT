#ifndef OCTREE_H
#define OCTREE_H

#include "point.h"
#include "vector.h"
#include "object.h"
#include "ray.h"
#include "lambertmaterial.h"
#include "box.h"
#include "primitive.h"
#include "intrec.h"

enum subNodes{NEB, NWB, SWB, SEB, NET, NWT, SWT, SET};

class Object;
class Node {
public:
	__device__ Node();
	__device__ Node(Point boundmin, Point boundmax, int id);
	__device__ Node(Point boundmin, Point boundmax, int octant, Node* parent, int id);
	static int nextId;
	int id;
	Node* parent;
	int octant;
	Node* nodes[8];
	Point bounds[2];
	int nObjects;
	const Object* object;

	// If possible inserts the object in the correct node in the octree
	__device__ int Insert(Object* object, int &id);
	// Removes the object in the node of the octree corresponding to the given location
	__device__ void Remove(Object* object);
	// Checks if a ray intersects with an object in the octree
	__device__ bool Intersect(const Ray &ray, IntRec& intRec) const;
	// Finds the next node in a pre-order stackless tree traversal
	__device__ Node* NextNode(const Node* current, const Ray &ray, float &closest) const;
	// Checks if a ray intersects with an internal node in the octree
	__device__ float NodeIntersect(const Ray &ray) const;
};

#endif