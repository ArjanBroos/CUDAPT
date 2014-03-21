#include "octree.h"
#include "point.h"
#include "primitive.h"
#include "light.h"
#include "object.h"
#include <iostream>
#include <cmath>

__device__ Node::Node() {
}

__device__ Node::Node(Point bounda, Point boundb, int id) : nObjects(0), object(nullptr), id(id), octant(-1), parent(nullptr)
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

int Node::nextId = 0;

__device__ Node::Node(Point bounda, Point boundb, int octant, Node* parent, int id) : nObjects(0), object(nullptr), id(id), octant(octant), parent(parent)
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

__device__ int Node::Insert(const Object* object, int &id)
{
	Point loc = (*object->loc);
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
				currentNode->nodes[NET] = new Node(Point((float)midx, (float)midy, (float)midz), Point(currentNode->bounds[1].x, currentNode->bounds[1].y, currentNode->bounds[1].z), NET, currentNode, id++);
			currentNode = currentNode->nodes[NET];
		}
		if(!east && north && top) {
			if(currentNode->nodes[NWT] == nullptr)
				currentNode->nodes[NWT] = new Node(Point(currentNode->bounds[0].x, (float)midy, (float)midz), Point((float)midx, currentNode->bounds[1].y, currentNode->bounds[1].z), NWT, currentNode, id++);
			currentNode = currentNode->nodes[NWT];
		}
		if(east && !north && top) {
			if(currentNode->nodes[SET] == nullptr)
				currentNode->nodes[SET] = new Node(Point((float)midx, currentNode->bounds[0].y, (float)midz), Point(currentNode->bounds[1].x, (float)midy, currentNode->bounds[1].z), SET, currentNode, id++);
			currentNode = currentNode->nodes[SET];
		}
		if(!east && !north && top) {
			if(currentNode->nodes[SWT] == nullptr)
				currentNode->nodes[SWT] = new Node(Point(currentNode->bounds[0].x, currentNode->bounds[0].y, (float)midz), Point((float)midx, (float)midy, currentNode->bounds[1].z), SWT, currentNode, id++);
			currentNode = currentNode->nodes[SWT];
		}
		if(east && north && !top) {
			if(currentNode->nodes[NEB] == nullptr)
				currentNode->nodes[NEB] = new Node(Point((float)midx, (float)midy, currentNode->bounds[0].z), Point(currentNode->bounds[1].x, currentNode->bounds[1].y, (float)midz), NEB, currentNode, id++);
			currentNode = currentNode->nodes[NEB];
		}
		if(!east && north && !top) {
			if(currentNode->nodes[NWB] == nullptr)
				currentNode->nodes[NWB] = new Node(Point(currentNode->bounds[0].x, (float)midy, currentNode->bounds[0].z), Point((float)midx, currentNode->bounds[1].y, (float)midz), NWB, currentNode, id++);
			currentNode = currentNode->nodes[NWB];
		}
		if(east && !north && !top) {
			if(currentNode->nodes[SEB] == nullptr)
				currentNode->nodes[SEB] = new Node(Point((float)midx, currentNode->bounds[0].y, currentNode->bounds[0].z), Point(currentNode->bounds[1].x, (float)midy, (float)midz), SEB, currentNode, id++);
			currentNode = currentNode->nodes[SEB];
		}
		if(!east && !north && !top) {
			if(currentNode->nodes[SWB] == nullptr)
				currentNode->nodes[SWB] = new Node(Point(currentNode->bounds[0].x, currentNode->bounds[0].y, currentNode->bounds[0].z), Point((float)midx, (float)midy, (float)midz), SWB, currentNode, id++);
			currentNode = currentNode->nodes[SWB];
		}
	}

	//Smalles node found, try to insert
	if(currentNode->object) {
		return -1;
	}
	currentNode->object = object;

	//Fix number of objects in parents
	while(currentNode->parent != nullptr) {
		currentNode->nObjects++;
		currentNode = currentNode->parent;
	}
	currentNode->nObjects++;
	return 1;
}

__device__ int Node::Remove(const Point &loc) {
	return 1;
}

__device__ bool Node::Intersect(const Ray &ray, IntRec& intRec) const {
	float temp;
	bool intersect = false;
	Node* current = NextNode(this, ray);
	while(current) {
		if(current->NodeIntersect(ray, temp) && temp < intRec.t) {
			if(current->object) {
				float distObj;
				if(current->object->Intersect(ray, distObj) && distObj < intRec.t) {
					intRec.t = distObj;
					intersect = true;
					if(current->object->type == PRIMITIVE) {
						intRec.prim = (Primitive*) current->object;
						intRec.light = NULL;
					} else {
						intRec.prim = NULL;
						intRec.light = (Light*) current->object;
					}
				}
			}
		}
		current = NextNode(current, ray);
	}
	return intersect;
}

__device__ Node* Node::NextNode(const Node* current, const Ray &ray) const{
	float temp;
	// Return the left most child node that intersects with the ray
	Node* child;
	for(int i = NEB; i <= SET; i++) {
		child = current->nodes[i];
		if(child && child->NodeIntersect(ray, temp)) {
			return child;
		}
	}

	// If it's a leaf node, find the next sibling or ascend and repeat
	while(current->parent) {
		Node* parent = current->parent;
		for(int i = current->octant + 1; i <= SET; i++) {
			if(parent->nodes[i] && parent->nodes[i]->NodeIntersect(ray, temp)) {
				//std::cout << "Check: " << current->parent->nodes[i] << std::endl;
				//std::cout << "Check: " << &ray << std::endl;
				return parent->nodes[i];
			}
		}
		current = current->parent;
	}
	//std::cout << "Check: " << std::endl;
	return nullptr;
}

__device__ bool Node::NodeIntersect(const Ray &ray, float &t) const {
	float tmin, tmax, tymin, tymax, tzmin, tzmax;
	//Check x-direction
	if (ray.d.x >= 0) {
		tmin = (bounds[0].x - ray.o.x) / ray.d.x;
		tmax = (bounds[1].x - ray.o.x) / ray.d.x;
	}
	else {
		tmin = (bounds[1].x - ray.o.x) / ray.d.x;
		tmax = (bounds[0].x - ray.o.x) / ray.d.x;
	}
	//Check y-direction
	if (ray.d.y >= 0) {
		tymin = (bounds[0].y - ray.o.y) / ray.d.y;
		tymax = (bounds[1].y - ray.o.y) / ray.d.y;
	}
	else {
		tymin = (bounds[1].y - ray.o.y) / ray.d.y;
		tymax = (bounds[0].y - ray.o.y) / ray.d.y;
	}
	//Compare to previous interval
	if ( (tmin > tymax) || (tymin > tmax) )
		return false;
	if (tymin > tmin)
		tmin = tymin;
	if (tymax < tmax)
		tmax = tymax;
	//Check z-direction
	if (ray.d.z >= 0) {
		tzmin = (bounds[0].z - ray.o.z) / ray.d.z;
		tzmax = (bounds[1].z - ray.o.z) / ray.d.z;
	}
	else {
		tzmin = (bounds[1].z - ray.o.z) / ray.d.z;
		tzmax = (bounds[0].z - ray.o.z) / ray.d.z;
	}
	//Compare to previous interval
	if ( (tmin > tzmax) || (tzmin > tmax) )
		return false;
	if (tzmin > tmin)
		tmin = tzmin;
	if (tzmax < tmax)
		tmax = tzmax;
	if ( (tmin < ray.maxt) && (tmax > ray.mint) && (tmax > 0)) {
		t = tmin;
		return true;
	}
	return false;
}