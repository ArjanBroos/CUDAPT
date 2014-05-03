#ifndef SCENE_H
#define SCENE_H

#include "cuda_inc.h"
#include "geometry.h"
#include "primitive.h"
#include "intrec.h"
#include "light.h"
#include "box.h"
#include "lambertmaterial.h"
#include "octree.h"
#include "plane.h"
#include "color.h"

#define MAX_PRIMITIVES	1024
#define MAX_OBJECTS	1024
#define MAX_PLANES	8
#define MAX_LIGHTS		1024

class Scene {
public:
	__device__ Scene();
	__device__ ~Scene();
	__device__ void	AddPrimitive(Primitive* primitive);
	__device__ void AddPlane(Primitive* plane);
	__device__ void AddLight(Light* light);
	// Adds object to the octree of the scene
	__device__ void AddObject(Object* object);
	__device__ void RemoveObject(Object* object);
	__device__ void IncreaseDayLight(float amount);
	__device__ void DecreaseDayLight(float amount);
	__device__ const Color GetDayLight() const;
	__device__ bool	Intersect(const Ray& ray, IntRec& intRec) const;
	__device__ bool	IntersectOctree(const Ray& ray, IntRec& intRec, bool& intersected) const;
	__device__ int GetNumberOfObjects() const;
	int			size;
	Node*		octree;

private:
	unsigned	primCounter;
	unsigned	lightCounter;
	Primitive**	primitives;
	Light**		lights;

	unsigned	planeCounter;
	unsigned	objectCounter;
	Primitive**	planes;
	Object**	objects;
	Color		dayLight;
};

#endif
