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

enum SubNodes{NEB, NWB, SWB, SEB, NET, NWT, SWT, SET};

class Object;
class Node {
public:
    Node();
    Node(Point boundmin, Point boundmax);
    Node(Point boundmin, Point boundmax, int octant, Node* parent);
    ~Node();
    std::string *name;
	Node* parent;
	int octant;
	Node* nodes[8];
	Point bounds[2];
	int nObjects;
    Object* object;

	// If possible inserts the object in the correct node in the octree
    int Insert(Object* object);
	// Removes the object in the node of the octree corresponding to the given location
    void Remove(Object* object);
	// Checks if a ray intersects with an object in the octree
    bool Intersect(const Ray &ray, IntRec& intRec) const;
	// Finds the next node in a pre-order stackless tree traversal with pruning via ray
    Node* NextNode(const Node* current, const Ray &ray, float &closest) const;
	// Finds the next leaf in a pre-order stackless tree traversal without pruning via ray
    Node* NextNode(const Node* current) const;
	// Finds the next leaf in a pre-order stackless tree traversal
    Node* NextLeaf(Node* current) const;
	// Checks if a ray intersects with an internal node in the octree
    float NodeIntersect(const Ray &ray) const;
	// Returns the number of objects in the tree
    int GetNumberOfObjects() const;
};

#endif
