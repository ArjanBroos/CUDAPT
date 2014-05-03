#ifndef INTREC_H
#define INTREC_H

#include "geometry.h"
#include "cuda_inc.h"
#include "math.h"
#include "primitive.h"
#include "light.h"

// Intersection record
class IntRec {
public:
	__device__ IntRec() : t(INFINITY), prim(nullptr), light(nullptr) {}
	float		t;
	Primitive*	prim;
	Light*		light;
};

#endif
