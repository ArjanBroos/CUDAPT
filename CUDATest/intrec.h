#ifndef INTREC_H
#define INTREC_H

#include "geometry.h"
#include "math.h"
#include "primitive.h"
#include "light.h"

// Intersection record
class IntRec {
public:
    IntRec() : t(INFINITY), prim(NULL), light(NULL) {}
	float		t;
	Primitive*	prim;
	Light*		light;
};

#endif
