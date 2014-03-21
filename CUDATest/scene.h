#pragma once

#include "cuda_inc.h"
#include "geometry.h"
#include "primitive.h"
#include "intrec.h"
#include "light.h"
#include "box.h"
#include "lambertmaterial.h"
#include "octree.h"
#include "plane.h"

#define MAX_PRIMITIVES	1024
#define MAX_OBJECTS	1024
#define MAX_PLANES	8
#define MAX_LIGHTS		32

class Scene {
public:
	__device__ Scene();
	__device__ ~Scene();
	__device__ void	AddPrimitive(Primitive* primitive);
	__device__ void AddPlane(Primitive* plane);
	__device__ void AddLight(Light* light);
	// Adds object to the octree of the scene
	__device__	void AddObject(Object* object);
	__device__ bool	Intersect(const Ray& ray, IntRec& intRec) const;
	__device__ bool	IntersectOctree(const Ray& ray, IntRec& intRec, bool& intersected) const;

private:
	unsigned	primCounter;
	unsigned	lightCounter;
	Primitive**	primitives;
	Light**		lights;

	unsigned	planeCounter;
	unsigned	objectCounter;
	Primitive**		planes;
	Object**	objects;
	int			nextId;
	//Point**				objectLocs;
	//Box**				objectShapes;
	//LambertMaterial**	objectMats;
	//Primitive**			objects;
	//Point**				lightLocs;
	//Box**				lightShapes;
	//AreaLight**			lights;
	Node				octree;
};