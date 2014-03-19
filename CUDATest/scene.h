#pragma once

#include "cuda_inc.h"
#include "geometry.h"
#include "primitive.h"
#include "intrec.h"
#include "light.h"

#define MAX_PRIMITIVES	1024
#define MAX_LIGHTS		32

class Scene {
public:
	__device__ Scene();
	__device__ ~Scene();
	__device__ void	AddPrimitive(Primitive* primitive);
	__device__ void AddLight(Light* light);
	__device__ bool	Intersect(const Ray& ray, IntRec& intRec) const;

private:
	unsigned	primCounter;
	unsigned	lightCounter;
	Primitive**	primitives;
	Light**		lights;
};