#ifndef RAY_H
#define RAY_H

#include "point.h"
#include "vector.h"
#include "math.h"
#include "cuda_inc.h"

// Represents a ray with an origin and direction through 3-dimensional space
class Ray {
public:
    Ray() {};
	Point		o;	// Origin
	Vector		d;	// Direction
	Vector		inv;
	int			sign[3];
	float		ior; // Index of refraction of the medium through which the ray is currently traveling

	// The ray is clamped between o + mint * d and o + maxt * d
	// To prevent intersection with a surface that this ray originated from
	float		mint;	// Minimum distance along ray for intersection checks
	float		maxt; // Maximum distance along ray for intersection checks

	// Initializes a ray with origin o and direction d. mint and maxt are optional
    Ray(const Point& o, const Vector& d, float ior = 1.f, float mint = 5e-4f, float maxt = INFINITY);
    //Ray(const Point& o, const Vector& d, float ior = 1.f, float mint = 2.f, float maxt = INFINITY);

	// Returns o + t*d
    Point		operator()(float t) const;
};

#endif
