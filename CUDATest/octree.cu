#include "octree.h"
#include "point.h"
#include "primitive.h"
#include "light.h"
#include "object.h"
#include <iostream>
#include "math.h"

__device__ Node::Node() {
}

__device__ Node::Node(Point bounda, Point boundb) : nObjects(0), object(nullptr), octant(-1), parent(nullptr)
{
	if(bounda < boundb) {
		bounds[0] = bounda;
		bounds[1] = boundb;
	} else {
		bounds[0] = boundb;
		bounds[1] = bounda;
	}
	for( int i = 0; i < 8; i++) 
		nodes[i] = nullptr;
}

__device__ Node::Node(Point bounda, Point boundb, int octant, Node* parent) : nObjects(0), object(nullptr), octant(octant), parent(parent)
{

	if(bounda < boundb) {
		bounds[0] = bounda;
		bounds[1] = boundb;
	} else {
		bounds[0] = boundb;
		bounds[1] = bounda;
	}
	for( int i = 0; i < 8; i++) 
		nodes[i] = nullptr;
}

__device__ int Node::Insert(Object* object)
{
	Point loc = (*object->loc);
	if(loc.x > bounds[1].x - 1 || loc.y > bounds[1].y - 1 || loc.z > bounds[1].z - 1) {
		return -1;
	}
	Node* currentNode = this;
	while( ! ( (fabsf(currentNode->bounds[1].x - currentNode->bounds[0].x - 1) < 1e-5f) 
		&& (fabsf(currentNode->bounds[1].y - currentNode->bounds[0].y - 1) < 1e-5f) 
		&& (fabsf(currentNode->bounds[1].z - currentNode->bounds[0].z - 1) < 1e-5f) ) )
	{
		//Is loc in this bounding box?
		if(loc < currentNode->bounds[0] || loc > currentNode->bounds[1])
			return 0;
		//Find leaf node
		int midx = (int) ((currentNode->bounds[0].x+currentNode->bounds[1].x)/2.0);
		int midy = (int) ((currentNode->bounds[0].y+currentNode->bounds[1].y)/2.0);
		int midz = (int) ((currentNode->bounds[0].z+currentNode->bounds[1].z)/2.0);
		bool east = loc.x >= midx;
		bool north = loc.y >= midy;
		bool top = loc.z >= midz;
		if(east && north && top) {
			if(currentNode->nodes[NET] == nullptr)
				currentNode->nodes[NET] = new Node(Point((float)midx, (float)midy, (float)midz), Point(currentNode->bounds[1].x, currentNode->bounds[1].y, currentNode->bounds[1].z), NET, currentNode);
			currentNode = currentNode->nodes[NET];
		}
		if(!east && north && top) {
			if(currentNode->nodes[NWT] == nullptr)
				currentNode->nodes[NWT] = new Node(Point(currentNode->bounds[0].x, (float)midy, (float)midz), Point((float)midx, currentNode->bounds[1].y, currentNode->bounds[1].z), NWT, currentNode);
			currentNode = currentNode->nodes[NWT];
		}
		if(east && !north && top) {
			if(currentNode->nodes[SET] == nullptr)
				currentNode->nodes[SET] = new Node(Point((float)midx, currentNode->bounds[0].y, (float)midz), Point(currentNode->bounds[1].x, (float)midy, currentNode->bounds[1].z), SET, currentNode);
			currentNode = currentNode->nodes[SET];
		}
		if(!east && !north && top) {
			if(currentNode->nodes[SWT] == nullptr)
				currentNode->nodes[SWT] = new Node(Point(currentNode->bounds[0].x, currentNode->bounds[0].y, (float)midz), Point((float)midx, (float)midy, currentNode->bounds[1].z), SWT, currentNode);
			currentNode = currentNode->nodes[SWT];
		}
		if(east && north && !top) {
			if(currentNode->nodes[NEB] == nullptr)
				currentNode->nodes[NEB] = new Node(Point((float)midx, (float)midy, currentNode->bounds[0].z), Point(currentNode->bounds[1].x, currentNode->bounds[1].y, (float)midz), NEB, currentNode);
			currentNode = currentNode->nodes[NEB];
		}
		if(!east && north && !top) {
			if(currentNode->nodes[NWB] == nullptr)
				currentNode->nodes[NWB] = new Node(Point(currentNode->bounds[0].x, (float)midy, currentNode->bounds[0].z), Point((float)midx, currentNode->bounds[1].y, (float)midz), NWB, currentNode);
			currentNode = currentNode->nodes[NWB];
		}
		if(east && !north && !top) {
			if(currentNode->nodes[SEB] == nullptr)
				currentNode->nodes[SEB] = new Node(Point((float)midx, currentNode->bounds[0].y, currentNode->bounds[0].z), Point(currentNode->bounds[1].x, (float)midy, (float)midz), SEB, currentNode);
			currentNode = currentNode->nodes[SEB];
		}
		if(!east && !north && !top) {
			if(currentNode->nodes[SWB] == nullptr)
				currentNode->nodes[SWB] = new Node(Point(currentNode->bounds[0].x, currentNode->bounds[0].y, currentNode->bounds[0].z), Point((float)midx, (float)midy, (float)midz), SWB, currentNode);
			currentNode = currentNode->nodes[SWB];
		}
	}

	//Smalles node found, try to insert
	if(currentNode->object) {
		return -1;
	}
	currentNode->object = object;
	object->parent = currentNode;

	//Fix number of objects in parents
	while(currentNode->parent != nullptr) {
		currentNode->nObjects++;
		currentNode = currentNode->parent;
	}
	currentNode->nObjects++;
	return 1;
}

__device__ void Node::Remove(Object* object) {
	if(object->GetType() == OBJ_PRIMITIVE && ((Primitive*)object)->GetShape()->GetType() == ST_PLANE)
		return;
	Node* currentNode, *previousNode;
	currentNode = object->parent;
	//Fix number of objects in parents
	while(currentNode->parent != nullptr) {
		currentNode->nObjects--;
		previousNode = currentNode;
		currentNode = currentNode->parent;
		if(previousNode->nObjects == 0) {
			delete currentNode->nodes[previousNode->octant];
			currentNode->nodes[previousNode->octant] = NULL;
		}
	}
	currentNode->nObjects--;
	delete object;
}

__device__ bool Node::Intersect(const Ray &ray, IntRec& intRec) const {
	float temp;
	bool intersect = false;
	Node* current = NextNode(this, ray, intRec.t);
	while(current) {
		if(current->object) {
			if(current->object->Intersect(ray, temp) && temp < intRec.t) {
				intRec.t = temp;
				intersect = true;
				if(current->object->GetType() == OBJ_PRIMITIVE) {
					intRec.prim = (Primitive*) current->object;
					intRec.light = NULL;
				}
				if(current->object->GetType() == OBJ_LIGHT) {
					intRec.prim = NULL;
					intRec.light = (Light*) current->object;
				}
			}
		}
		current = NextNode(current, ray, intRec.t);
	}
	return intersect;
}

__device__ Node* Node::NextNode(const Node* current, const Ray &ray, float &closest) const{
	// Return the left most child node that intersects with the ray
	Node* node;
	for(int i = NEB; i <= SET; i++) {
		node = current->nodes[i];
		if(node && node->NodeIntersect(ray) < closest) {
			return node;
		}
	}

	// If it's a leaf node, find the next sibling or ascend and repeat
	while(current->parent) {
		node = current->parent;
		for(int i = current->octant + 1; i <= SET; i++) {
			Node* node2 = node->nodes[i];
			if(node2 && node2->NodeIntersect(ray) < closest) {
				return node2;
			}
		}
		current = current->parent;
	}
	return nullptr;
}

__device__ Node* Node::NextNode(const Node* current) const {
	// Return the left most child node that intersects with the ray
	Node* node;
	for(int i = NEB; i <= SET; i++) {
		node = current->nodes[i];
		if(node) {
			return node;
		}
	}

	// If it's a leaf node, find the next sibling or ascend and repeat
	while(current->parent) {
		node = current->parent;
		for(int i = current->octant + 1; i <= SET; i++) {
			Node* node2 = node->nodes[i];
			if(node2) {
				return node2;
			}
		}
		current = current->parent;
	}
	return nullptr;
}

__device__ Node* Node::NextLeaf(Node* current) const {
	current = NextNode(current);
	while(current) {
		if(current->object) {
			return current;
		}
		current = NextNode(current);
	}
	return nullptr;
}

__device__ float Node::NodeIntersect(const Ray &ray) const {
	float tmin, tmax, tminn, tmaxn;

	tmin = (bounds[ray.sign[0]].x - ray.o.x) * ray.inv.x;
	tmax = (bounds[1-ray.sign[0]].x - ray.o.x) * ray.inv.x;
	tminn = (bounds[ray.sign[1]].y - ray.o.y) * ray.inv.y;
	tmaxn = (bounds[1-ray.sign[1]].y - ray.o.y) * ray.inv.y;

	//Compare to previous interval
	if ( (tmin > tmaxn) || (tminn > tmax) )
		return INFINITY;
	if (tminn > tmin)
		tmin = tminn;
	if (tmaxn < tmax)
		tmax = tmaxn;

	tminn = (bounds[ray.sign[2]].z - ray.o.z) * ray.inv.z;
	tmaxn = (bounds[1-ray.sign[2]].z - ray.o.z) * ray.inv.z;
	//Compare to previous interval
	if ( (tmin > tmaxn) || (tminn > tmax) )
		return INFINITY;
	if (tminn > tmin)
		tmin = tminn;
	if (tmaxn < tmax)
		tmax = tmaxn;
	if(tmax < 0)
		return INFINITY;
	if ( (tmin < ray.maxt) && (tmax > ray.mint) && (tmax > 0)) {
		return tmin;
	}
	return INFINITY;
}

__device__ int Node::GetNumberOfObjects() const {
	return nObjects;
}